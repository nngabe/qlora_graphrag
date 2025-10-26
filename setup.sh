#!/bin/bash

# please run the following commands before executing this script:
#
# git clone https://github.com/nngabe/qlora_graphrag.git
# cd qlora_graphrag/

conda init
conda create -n panther python=3.11
conda activate panther
python -m pip install -r requirements.txt 

sed -i '7c from langchain_text_splitters import RecursiveCharacterTextSplitter' /venv/panther/lib/python3.11/site-packages/stark_qa/tools/process_text.py
