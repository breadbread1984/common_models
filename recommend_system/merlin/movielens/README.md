# Introduction

this repo implements DLRM training

# Usage

## Download dataset

```shell
python3 create_datasets.py [--output_dir dataset]
```

## Train model

```shell
python3 train.py
```

## Serving

```shell
python3 convert2torchscript.py
```
