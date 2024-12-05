# Introduction

this project demos how the LLM is supervisedly pretrained.

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## Download dataset

```shell
python3 create_datasets.py
```

## Training

```shell
deepspeed --num_gpus=<tensor parallelism number> train.py [other args]
```
