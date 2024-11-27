# Introduction

this project demonstrates how to create customized dataset and train two-tower model on the dataset

# Usage

## Download dataset

download [Alibaba Click and Convresion Prediction](https://tianchi.aliyun.com/dataset/408) and uncompress the the .tar.gz file with the following commands

```shell
mkdir -p aliccp/train
mkdir -p aliccp/test
tar xzvf sample_train.tar.gz --directory=aliccp/train
tar xzvf sample_test.tar.gz --directory=aliccp/test
```

## Convert raw dataset to parquet format

```shell
python3 create_datasets.py --dataset aliccp
```

## Training DLRM

```shell
python3 train.py --dataset aliccp
```

## Deployment

### export model to torchscript

```shell
python3 export.py --ckpt <path/to/ckpt> --output aliccp_click_model --dataset aliccp
```
