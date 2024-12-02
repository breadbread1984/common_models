# Introduction

this project demos how to extract knowledge graph from text file

# Usage

## Install prerequisite

```shell
python3 -m pip install requirements.txt
```

## Launch text-generate-inference server

```shell
docker pull ghcr.io/huggingface/text-generation-inference
model=Qwen/Qwen2.5-7B-Instruct
docker run --gpus 0,1,2,3 --shm-size 1g -p 8080:80 -v /home/xieyi/raid/huggingface:/data ghcr.io/huggingface/text-generation-inference --model-id $model --max-input-length 52207 --max-batch-prefill-tokens 52207 --max-total-tokens 131072 --max-batch-size 32 --num-shard 4
```

## Extract knowledge graph

```shell
python3 main.py --input_dir <input/directory> --tgi_host <tgi/host> --neo4j_host <neo4j/host> --neo4j_user <neo4j/user> --neo4j_password <neo4j/password> --neo4j_db <neo4j/db>
```

