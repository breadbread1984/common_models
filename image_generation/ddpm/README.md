# Introduction

this project implement DDPM trained on CIFAR10 dataset

# Usage

## Install prerequisite

```shell
python3 -m pip install requirements.txt
```

## Training

```shell
torchrun --nproc_per_node  1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8082 train.py
```
