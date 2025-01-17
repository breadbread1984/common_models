# Introduction

this project demos how to deploy model with tritonserver

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
docker pull nvcr.io/nvidia/tritonserver:24.11-py3  # for deployment
docker pull nvcr.io/nvidia/tensorrt:24.11-py3  # for tensorrt engine generation
```

## Convert model

### Torch Model

use detection model trained in [object_detection](../object_detection)

```shell
python3 convert2torchscript.py --ckpt <path/to/checkpoint> --model (fasterrcnn_resnet50_fpn|fasterrcnn_mobilenet_v3_large_fpn|fasterrcnn_mobilenet_v3_large_320_fpn|fcos_resnet50_fpn|retinanet_resnet50_fpn|ssd300_vgg16|ssdlite320_mobilenet_v3_large|maskrcnn_resnet50_fpn|keypointrcnn_resnet50_fpn) --weights_backbone (ResNet50_Weights.IMAGENET1K_V1|VGG16_Weights.IMAGENET1K_FEATURES|MobileNet_V3_Large_Weights.IMAGENET1K_V1) --classnum <class/number> --type (trace|script) --device (cuda|cpu)
```

**NOTE: you can only inference the torchscript with the device which you use for generating torchscript!**

**NOTE: generate torchscript with trace method is preferable!**

### Tensorflow Model

```shell
WRAPT_DISABLE_EXTENSIONS=true python3 convert2savedmodel.py --model (convnext|densenet|efficientnet|inception|mobilenet)
```

### TensorRT Model

Upon generating torch and tensorflow model successfully, their onnx model can be generated by

```shell
python3 convert2onnx.py --type (pt|tf) --output <onnx/path> --device (cpu|cuda)
```

convert onnx into TensorRT engine

```shell
python3 convert2trt.py --type (pt|tf) --input <onnx/path> --output <engine/path>
```

## Deployment with Triton Server

```shell
python3 deploy.py --model models --host <host> --port <port>
```

## Test Service

### Test Torch Model

```shell
python3 test_torch.py --host <host> --port <port> --method network
```

test can be also performed locally

```shell
python3 test_torch.py --model models/torch_model/1/model.pt --device (cuda|cpu) --method local
```

**NOTE: you can only inference the torchscript with the device which you use for generating torchscript!**

### Test Tensorflow Model

```shell
python3 test_tensorflow.py --host <host> --port <port> --method network
```

test can be also performed locally

```shell
python3 test_tensorflow.py --model models/tensorflow_model/1/model.savedmodel --method local
```
