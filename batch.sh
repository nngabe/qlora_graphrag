#!/bin/bash

python train.py --checkpointing  --device cuda --freeze_llm False --llama_version llama3.2-3b --use_lora True --use_quantization True
python train.py --checkpointing  --device cuda --freeze_llm False --llama_version llama3.2-3b --use_lora True --use_quantization False
python train.py --checkpointing  --device cuda --freeze_llm False --llama_version llama3.2-3b --use_lora False --use_quantization False
python train.py --checkpointing  --device cuda --freeze_llm False --llama_version llama3.1-8b --use_lora False --use_quantization False
