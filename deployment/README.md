# Introduction

this project demos how to deploy model with tritonserver

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## Convert model

### Torch Model

```shell
docker pull nvcr.io/nvidia/tritonserver:24.11-pyt-python-py3
```

use detection model trained in [object_detection](../object_detection)

```shell
python3 convert2torchscript.py --ckpt <path/to/checkpoint> --output <path/to/output> --model (fasterrcnn_resnet50_fpn|fasterrcnn_mobilenet_v3_large_fpn|fasterrcnn_mobilenet_v3_large_320_fpn|fcos_resnet50_fpn|retinanet_resnet50_fpn|ssd300_vgg16|ssdlite320_mobilenet_v3_large|maskrcnn_resnet50_fpn|keypointrcnn_resnet50_fpn) --weights_backbone (ResNet50_Weights.IMAGENET1K_V1|VGG16_Weights.IMAGENET1K_FEATURES|MobileNet_V3_Large_Weights.IMAGENET1K_V1) --classnum <class/number> --type (trace|script)
```

### Tensorflow Model

```shell
docker pull nvcr.io/nvidia/tritonserver:24.11-tf2-python-py3
```

```shell
WRAPT_DISABLE_EXTENSIONS=true python3 convert2savedmodel.py --model (convnext|densenet|efficientnet|inception|mobilenet) --output <path/to/output>
```

## Deployment

```shell
python3 deploy.py --model <path/to/model> --host <host> --port <port>
```

