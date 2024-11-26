# Introduction

this project demonstrates how to create customized dataset and train two-tower model on the dataset

# Usage

## Download dataset

download [Alibaba Click and Convresion Prediction](https://tianchi.aliyun.com/dataset/408) and uncompress the the .tar.gz file as the following paths

| file |  uncompressed path |
|------|--------------------|
|sample_train.tar.gz | dataset/train |
|sample_test.tar.gz | dataset/test |

you may use the following command

```shell
mkdir -p dataset/train
mkdir -p dataset/test
tar xzvf sample_(train|test).tar.gz --directory=dataset/(train|test)
```

```shell
python3 create_datasets.py --dataset dataset
```

