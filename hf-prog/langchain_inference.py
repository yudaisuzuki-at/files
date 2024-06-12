import sys
import torch
from langchain import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import  PeftModel
from argparse import ArgumentParser
import time
from omegaconf import OmegaConf
stop_token_ids = None

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
            "--inf_config",
            "-c",
            type=str,
            default="./config/hf_config.yaml"
            )
    args = parser.parse_args()
    hfconfig = OmegaConf.load(args.inf_config)
    global stop_token_ids
    
    #------------------------------------------------
    # 使用するGPU
    device_map = hfconfig.device.device_map or 'auto'
    print(f'Using {device_map=}', file=sys.stderr)
    #------------------------------------------------        
    # model_dtypeの決定。略称から。
    model_dtype = get_dtype(hfconfig.device.model_dtype)
    print(f'Using {model_dtype=}')
    #------------------------------------------------    
    # 設定読み込み
    from_pretrained_kwargs = {'trust_remote_code': hfconfig.device.trust_remote_code,}
    try:
        config = AutoConfig.from_pretrained(hfconfig.data.modelname,
                                            **from_pretrained_kwargs)

        major, minor = torch.cuda.get_device_capability()
        if hfconfig.tokenize.attn_impl is not None and hasattr(config, 'attn_config'):
            config.attn_config['attn_impl'] = hfconfig.tokenize.attn_impl
        if hfconfig.inference.max_seq_len is not None and hasattr(config, 'max_seq_len'):
            config.max_seq_len = hfconfig.inference.max_seq_len
    except Exception as e:
        raise RuntimeError(e.args) from e
    #------------------------------------------------
    # トークナイザー読み込み
    print('\nLoading HF tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(hfconfig.data.modelname, use_fast=hfconfig.tokenize.use_fast,**from_pretrained_kwargs)

    if tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
    #------------------------------------------------
    # HF版モデルロード
    print(f'Loading HF model with dtype={model_dtype}...', file=sys.stderr)
    try:
        model = AutoModelForCausalLM.from_pretrained(hfconfig.data.modelname,
                                                     config=config,
                                                     torch_dtype=model_dtype,
                                                     device_map=device_map,
                                                     **from_pretrained_kwargs)
        model.eval()
        print(f'n_params={sum(p.numel() for p in model.parameters())}', file=sys.stderr)
    except Exception as e:
        raise RuntimeError(e.args) from e
    #------------------------------------------------
    # LoRAモデルのロード
    if hfconfig.data.lora_path:
        print(f'Loading Peft model with dtype={model_dtype}...', file=sys.stderr)
        try:
            model = PeftModel.from_pretrained(model,
                                              hfconfig.data.lora_path,
                                              torch_dtype=model_dtype,
                                              device_map=device_map,
                                              **from_pretrained_kwargs)
            model.eval()
        except Exception as e:
            raise RuntimeError(e.args) from e

    #------------------------------------------------
    # パイプラインの作成
    pipeline_kwargs = dict(
        min_new_tokens=hfconfig.inference.min_new_tokens,
        max_new_tokens=hfconfig.inference.max_new_tokens,
        temperature=hfconfig.inference.temperature,
        top_p=hfconfig.inference.top_p,
        top_k=hfconfig.inference.top_k,
        repetition_penalty=hfconfig.inference.repetition_penalty,
        no_repeat_ngram_size=hfconfig.inference.no_repeat_ngram_size,
        use_cache=hfconfig.inference.use_cache,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if hfconfig.inference.num_beams and hfconfig.inference.num_beams > 1:
        pipeline_kwargs.update(
            num_beams=hfconfig.inference.num_beams,
            num_beam_groups=hfconfig.inference.num_beam_groups if hfconfig.inference.num_beam_groups else hfconfig.inference.num_beams,
            diversity_penalty=hfconfig.inference.diversity_penalty,
        )
    else:
        pipeline_kwargs.update(
            do_sample=hfconfig.inference.do_sample if hfconfig.inference.temperature > 0. else False
        )
    pipe = pipeline(
        task=hfconfig.inference.task,
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs
    )

    llm = HuggingFacePipeline(
        pipeline=pipe,
    )

    # プロンプトのテンプレートを定義
    prompt = PromptTemplate.from_template(hfconfig.data.system_prompt+"{text}")

    # LLM のチェーンを作成
    chain = LLMChain(llm=llm, prompt=prompt)

    text = '<|user|>' + f'{hfconfig.data.prompt}' + '<|endofuser|>'
    print('Generating responses...')

    print(f'------- Prompt')
    print(f'{text}')

    print(f'------- Generating Response...')
    start = time.time()
    result = chain.run(text=text)

    end = time.time()

    ids = tokenizer.encode(result,hfconfig.tokenize.add_special_tokens)
    total_tokens = len(ids)

    print(f'------- Response: ({total_tokens} tokens.)')
    print(f'{result}')

    print(f'------- Done. generated {total_tokens} tokens. {end-start} sec. {total_tokens/(end-start)} tokens/sec.' )


# 省略引数から置き換え
def get_dtype(dtype: str):
    if dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f'dtype {dtype} is not supported. ' +\
            f'We only support fp32, fp16, and bf16 currently')

if __name__ == '__main__':
    main()
