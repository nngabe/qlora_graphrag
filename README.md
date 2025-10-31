# LLM Finetuning: GraphRAG with GNN+LLM architecture on STaRK-Prime Knowledge Graph

This repo contains experiments for Knowledge Graph Retrieval (GraphRAG) with a GNN+LLM architecture for Q&A on the STaRK-Prime benchmark dataset. We use the base GNN+LLM architecture [G-Retriever](https://arxiv.org/abs/2402.07630), and [STaRK-Prime KG Q&A](https://stark.stanford.edu/dataset_prime.html).

This work is based on [neo4j-product-examples/neo4j-gnn-llm-example](https://github.com/neo4j-product-examples/neo4j-gnn-llm-example.git). We extend these results as follows:

1. Add support for model quantization and mixed precision QLoRA using `peft` and `bitsandbytes`.
2. Enable multi-gpu training with DDP/FSDP using `accelerate`.
3. Add `bitsandbytes` 8-bit optimizers as defaults: `AdamW8Bit` or `PagedAdamW8Bit`. 
4. Introduce an MPNN architecture with edge convolution and multiple aggregations to improve encoding of textualized Knowledge Graphs in `MPNN.py`.
5. With a greatly reduced memory footprint and improved architecture, we finetune an MPNN+LLM with Llama-70B and QLoRA, significantly improving Q&A performance on STaRK-Prime Q&A benchmark.

## Architecture Overview

- RAG on large knowledge graphs that require multi-hop retrieval and reasoning, beyond node classification and link prediction.
- General, extensible 2-part architecture: KG Retrieval & GNN+LLM.
- Efficient, stable inference time and output for real-world use cases.

For more details, see [neo4j-product-examples/neo4j-gnn-llm-example](https://github.com/neo4j-product-examples/neo4j-gnn-llm-example.git).

## Installation

First, set up this repository: 

```
git clone https://github.com/nngabe/qlora_graphrag.git
cd qlora_graphrag/
```

Then install the necessary dependencies and datasets:

```
source setup.sh
```

To download Llama models, you will need to login to huggingface using
```
hf auth login --token YOUR_HF_TOKEN_WITH_LLAMA_ACCESS
```

Lastly, to reproduce the results of the experiments run:
```
python -u single.py --device cuda --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm --lora_rank 4096 --lora_alpha 2048 --epochs 4
python -u single.py --device cuda --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm --lora_rank 4096 --lora_alpha 2048 --epochs 4 --gnn mpnn
python -u single.py --device cuda --llama_version llama3.3-70b --use_lora --use_quantization --freeze_llm --lora_rank 512 --lora_alpha 256 --eval_batch_size 4 --epochs 4 --paged_adamw --gnn mpnn
```
or on multi-gpu nodes:
```
accelerate launch --config_file configs/fsdp.yaml multi.py --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm --lora_rank 4096 --lora_alpha 2048 --epochs 4
accelerate launch --config_file configs/fsdp.yaml multi.py --llama_version llama3.1-8b --use_lora --use_quantization --freeze_llm --lora_rank 4096 --lora_alpha 2048 --epochs 4 --gnn mpnn
accelerate launch --config_file configs/fsdp.yaml multi.py --llama_version llama3.3-70b --use_lora --use_quantization --freeze_llm --lora_rank 512 --lora_alpha 256 --eval_batch_size 4 --epochs 4 --paged_adamw --gnn mpnn
```

