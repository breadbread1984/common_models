# Introduction

this project demos how retrievel augmented generation works

# Usage

## Install prerequisite

```shell
python3 -m pip install requirements.txt
```

## Launch Neo4j

```shell
docker pull neo4j:enterprise-bullseye
docker run -d --publish=7474:7474 --publish=7687:7687 \
           --volume=$HOME/neo4j/data:/data \
           --name neo4j-apoc \
           -e NEO4J_apoc_export_file_enabled=true \
           -e NEO4J_apoc_import_file_enabled=true \
           -e NEO4J_apoc_import_file_use__neo4j__config=true \
           -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
           --privileged --shm-size 12G -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes --cpus=32 --memory=128G neo4j
```

create a database for receiving text chunks

## Launch text-generation-inference server

```shell
docker pull ghcr.io/huggingface/text-generation-inference
model=Qwen/Qwen2.5-7B-Instruct
docker run --gpus 0,1,2,3 --shm-size 1g -p 8080:80 -v /home/xieyi/raid/huggingface:/data ghcr.io/huggingface/text-generation-inference --model-id $model --max-input-length 52207 --max-batch-prefill-tokens 52207 --max-total-tokens 131072 --max-batch-size 32 --num-shard 4
```

## Split and Index text chunks to Neo4j

```shell
python3 load_text.py --input_dir <input/directory> --tgi_host <tgi/host> --neo4j_host <neo4j/host> --neo4j_user <neo4j/user> --neo4j_password <neo4j/password> --neo4j_db <neo4j/db>
```

