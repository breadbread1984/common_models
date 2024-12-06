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
deepspeed --include localhost:3,4,5,6 train.py --tp <tp num> --pp <pp num> --dp <dp num>
```
