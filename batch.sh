#!/bin/bash

python train.py --checkpointing  --device cuda --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm --init_lora_weights eva
python train.py --checkpointing  --device cuda --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm --init_lora_weights olora
python train.py --checkpointing  --device cuda --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm
python train.py --checkpointing  --device cuda --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm --lora_rank 64 --lora_alpha 32
python train.py --checkpointing  --device cuda --llama_version llama3.2-3b --use_lora --use_quantization --freeze_llm --lora_rank 64 --lora_alpha 32
python train.py --checkpointing  --device cuda --llama_version llama3.2-3b --use_lora --use_quantization --freeze_llm --lora_rank 128 --lora_alpha 64
python train.py --checkpointing  --device cuda --llama_version llama3.2-3b --use_lora --use_quantization --freeze_llm
python train.py --checkpointing  --device cuda --llama_version llama3.2-3b --use_lora --freeze_llm
python train.py --checkpointing  --device cuda --llama_version llama3.2-1b 
