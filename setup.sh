# please run the following commands before executing this script:
#
# git clone https://github.com/nngabe/qlora_graphrag.git
# cd qlora_graphrag/
#source /opt/miniforge3/etc/profile.d/conda.sh

#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expendable_segments:True"

alias lhtr='ls -lhtr'

rm -rf gretriever/
git clone https://github.com/nngabe/gretriever.git

conda create -n panther python=3.11
conda activate panther
python -m pip install -r requirements.txt 
python -m pip install -U transformers peft accelerate bitsandbytes
sed -i '7c from langchain_text_splitters import RecursiveCharacterTextSplitter' /venv/panther/lib/python3.11/site-packages/stark_qa/tools/process_text.py

#MAX_JOBS=48 python -m pip -v install flash-attn --no-build-isolation # adjust MAX_JOBS for your system 
