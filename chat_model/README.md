# Introduction

this project implements a basic chatbot with langchain

# Usage

## install prerequisites

```shell
python3 -m pip install -r requirements.txt
```

## deploy text-generate-inference

```shell
model=Qwen/Qwen2.5-7B-Instruct
docker pull ghcr.io/huggingface/text-generation-inference
docker run --gpus all --shm-size 1g -p 8080:80 -v <home>/.cache/huggingface:/data ghcr.io/huggingface/text-generation-inference --model-id $model
```

## start chatbot

```shell
python3 main.py
```

visit the chatbot by url ***http://localhost:8081***
