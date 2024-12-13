# Introduction

this project implements python3 scripts for calling merline

# Usage

## Install prerequisite

### Install and launch Merlin-tensorflow if you use tensorflow implement

```shell
docker pull nvcr.io/nvidia/merlin/merlin-tensorflow:nightly
docker run --gpus all  --rm -td -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-tensorflow:nightly
```

### Install and launch Merlin-pytorch if you use torch implement

```shell
docker pull nvcr.io/nvidia/merlin/merlin-pytorch:nightly
docker run --gpus all  --rm -td -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-pytorch:nightly
```


