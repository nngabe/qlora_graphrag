#!/bin/bash

# please run the following commands before executing this script:
#
# git clone https://github.com/nngabe/qlora_graphrag.git
# cd qlora_graphrag/


rm -rf gretriever/
git clone https://github.com/nngabe/gretriever.git
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg
echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j=1:2025.09.0
neo4j start
cp /var/lib/neo4j/products/neo4j-g* /var/lib/neo4j/plugins/
sudo bash -c "echo 'dbms.security.procedures.allowlist=gds.*' >> /etc/neo4j/neo4j.conf"
/bin/neo4j-admin dbms set-initial-password FGtwLduAEhs3U4j
neo4j restart

conda create -n stark python=3.11
conda activate stark
python -m pip install -r requirements.txt 
cd gretriever/data-loading/
python emb_download.py --dataset prime --emb_dir emb/
python load_data.py 
