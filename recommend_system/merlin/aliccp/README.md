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

## Training TwoTower

```shell
python3 train_twotower.py --dataset aliccp
```

## Training DLRM

```shell
python3 train_dlrm.py --dataset aliccp
python3 export_dlrm.py --dataset aliccp --ckpt dlrm_ckpt
```

or

```shell
python3 train_dlrm_torch.py --dataset aliccp
python3 export_dlrm_torch.py --dataset aliccp --ckpt dlrm_ckpt
```

## Deployment

### extract item feature with trained two tower model

```shell
python3 extract_item_feature.py --dataset aliccp
python3 extract_user_feature.py --dataset aliccp
```

recommend items to users with extracted item features by measuring the inner product between a user feature and item features

### export model to torchscript

```shell
python3 export_dlrm.py --ckpt <path/to/ckpt> --output aliccp_click_model --dataset aliccp
```
