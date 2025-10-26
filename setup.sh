#!/bin/bash

# please run the following commands before executing this script:
#
# git clone https://github.com/nngabe/qlora_graphrag.git
# cd qlora_graphrag/
#source /opt/miniforge3/etc/profile.d/conda.sh

rm -rf gretriever/
git clone https://github.com/nngabe/gretriever.git

conda create -n panther python=3.11
conda activate panther
python -m pip install -r requirements.txt 
python -m pip install -U transformers peft accelerate bitsandbytes

sed -i '7c from langchain_text_splitters import RecursiveCharacterTextSplitter' /venv/panther/lib/python3.11/site-packages/stark_qa/tools/process_text.py
