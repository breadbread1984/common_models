#!/usr/bin/python3

import numpy as np
import torch
from torchvision.transforms import PILToTensor
from torchvision.datasets import CelebA
from facenet_pytorch import MTCNN, InceptionResnetV1
from db import DB

class Recognition(object):
  def __init__(self, device = 'cuda'):
    self.db = DB.create(hidden_dim = 1024)
    self.mtcnn = MTCNN(image_size = 160, margin = 0, min_face_size = 20,
                       thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                       device = device).eval()
    self.resnet = InceptionResnetV1(classify = False, pretrained = 'vggface2').to(device).eval()
    self.device = device
    self.labels = None
  def load_celeba(self,):
    trainset = CelebA(root = 'celeba', split = 'train', target_type = 'identity', download = True)
    batch = list()
    labels = list()
    for img, label in dataset:
      x = PILToTensor(img).to(self.device) # rgb
      x_aligned, prob = self.mtcnn(x, return_prob = True)
      if x_aligned is None: continue
      batch.append(x_aligned)
      labels.append(label)
      if len(batch) == 64:
        aligned = torch.stack(batch).to(device)
        embeddings = self.resnet(aligned).detach().cpu().numpy()
        self.db.add(embeddings)
        batch = list()
    if len(batch):
      aligned = torch.stack(batch).to(device)
      embeddings = self.resnet(aligned).detach().cpu().numpy()
      self.db.add(embeddings)
    self.labels = torch.stack(labels).detach().cpu().numpy()

if __name__ == "__main__":
  recog = Recognition()
  recog.load_celeba()
