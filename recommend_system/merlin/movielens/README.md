# Introduction

this repo implements DLRM training

# Usage

## Download dataset

```shell
python3 create_datasets.py [--output_dir dataset]
```

## Train model and save pipeline

### Tensorflow version

```shell
python3 train_tensorflow.py
```

### Pytorch version

```shell
python3 train_torch.py
```

***NOTE: due to imcompleted torch implement of DLRMModel, torch pipeline cannot be saved!***

## Serving

```shell
tritonserver --model-repository=pipeline_tf --http-port=8081 --grpc-port=8082 --metrics-port=8083
```
