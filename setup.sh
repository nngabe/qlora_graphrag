# please run the following commands before executing this script:
#
# git clone https://github.com/nngabe/qlora_graphrag.git
# cd qlora_graphrag/

rm -rf gretriever/
git clone https://github.com/nngabe/gretriever.git

conda create -n stark python=3.11
conda activate stark
python -m pip install -r requirements.txt 
python -m pip install -U transformers peft accelerate bitsandbytes
sed -i '7c from langchain_text_splitters import RecursiveCharacterTextSplitter' /venv/stark/lib/python3.11/site-packages/stark_qa/tools/process_text.py
python get_precomputed_kg.py # get precomputed KG with default config: stark_qa_v0_0

#MAX_JOBS=48 python -m pip -v install flash-attn --no-build-isolation # adjust MAX_JOBS for your system 
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expendable_segments:True"
