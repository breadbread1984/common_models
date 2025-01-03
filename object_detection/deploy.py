#!/usr/bin/python3

import numpy as np
import torch
from torchvision.models import get_model, get_weight
from torchvision.transforms.v2 import Compose, ToTensor, ToDtype

class Detection(object):
  def __init__(self, model = 'retinanet_resnet50_fpn', backbone_weight = 'ResNet50_Weights.IMAGENET1K_V1', n_classes = 4, ckpt_path = 'checkpoint.pth', device = 'cuda'):
    assert device in {'cuda', 'cpu'}
    ckpt = torch.load(ckpt_path, map_location = device)
    state_dict = ckpt['model']
    self.model = get_model(model, weights = None, weights_backbone = backbone_weight, num_classes = n_classes).to(device)
    self.model.load_state_dict(state_dict)
    self.model.eval()
    self.trans = Compose([
      ToTensor(),
      ToDtype(torch.float, scale = True),
    ])
  def detect(self, x):
    assert type(x) is np.ndarray
    # 1) convert to RGB
    x = x[:,:,::-1]
    # 2) add batch
    x = np.ascontiguousarray(x)
    # 3) preprocess
    x = self.trans(x).unsqueeze(dim = 0).to(next(self.model.parameters()).device)
    # 4) predict
    outputs = self.model(x)
    outputs = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs]
    return outputs[0]

if __name__ == "__main__":
  detection = Detection(device = 'cpu')
  import cv2
  img = cv2.imread('datasets/safetyhelmet/VOC2028/JPEGImages/PartB_00161.jpg')
  objs = detection.detect(img)
  boxes, scores, labels = objs['boxes'], objs['scores'], objs['labels']
  for box, score, label in zip(boxes, scores, labels):
    if score < 0.5: continue
    color = (255 if label == 0 else 0,255 if label == 1 else 0,255 if label == 2 else 0)
    cv2.rectangle(img, tuple(box[:2].astype(np.int32).tolist()), tuple(box[2:].astype(np.int32).tolist()), color, 2, 1)
  cv2.imwrite('output.png', img)
  cv2.imshow('', img)
  cv2.waitKey()
