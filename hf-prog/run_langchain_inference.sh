#!/bin/bash

# 使うGPU番号を CUDA_VISIBLE_DEVICES に指定して実行するのが最も確実です。
CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES="0"
prog='../src/langchain_inference.py'

# INPUT-LLM PATH
configfile='../src/config/hf_config.yaml'

python3 $prog \
	-c $configfile


