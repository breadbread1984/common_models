# Introduction

this project demonstrates how to create customized dataset and train two-tower model on the dataset

# Usage

## Download dataset

download [Alibaba Click and Convresion Prediction](https://tianchi.aliyun.com/dataset/408) and uncompress the the .tar.gz file with the following commands

```shell
mkdir aliccp
tar xzvf sample_train.tar.gz --directory=aliccp
tar xzvf sample_test.tar.gz --directory=aliccp
```

```shell
python3 create_datasets.py --dataset aliccp
```

