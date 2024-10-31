#!/usr/bin/python3

import numpy as np
import torch
from torchvision.models import get_model, get_weight

class Detection(object):
  def __init__(self, model = 'retinanet_resnet50_fpn', backbone_weight = 'ResNet50_Weights.IMAGENET1K_V1', n_classes = 4, ckpt_path = 'checkpoint.pth', device = 'cuda'):
    assert device in {'cuda', 'cpu'}
    ckpt = torch.load(ckpt_path, map_location = device)
    state_dict = ckpt['model']
    self.model = get_model(model = model, weights_backbone = backbone_weight, num_classes = n_classes).to(device)
    self.model.load_state_dict(state_dict)
    self.model.eval()
    self.trans = get_weight(backbone_weight).transforms()
    self.device = device
  def detect(self, x):
    assert type(x) is np.ndarray
    # 1) convert to RGB
    x = x[:,:,::-1]
    # 2) add batch
    x = torch.from_numpy(np.expand_dims(x, axis = 0), dtype = torch.float32).to(device)
    # 3) preprocess
    x = self.trans(x)
    # 4) predict
    outputs = self.model(x)
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    import pdb; pdb.set_trace()

if __name__ == "__main__":
  detection = Detection()

