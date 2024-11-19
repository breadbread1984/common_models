#!/usr/bin/python3

import numpy as np
import torch
from torch import nn

class Generator(nn.Module):
  def __init__(self, img_size, latent_dim = 100):
    super(Generator, self).__init__()
    def block(in_feat, out_feat, normalize = True):
      layers = [nn.Linear(in_feat, out_feat)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace = True))
      return layers
    self.model = nn.Sequential(
      *block(latent_dim, 128, normalize = False),
      *block(128,256),
      *block(256,512),
      *block(512, 1024),
      nn.Linear(1024, int(np.prod([3, img_size, img_size]))),
      nn.Tanh()
    )
    self.img_size = img_size
  def forward(self, z):
    img = self.model(z)
    img = torch.reshape(img, (img.shape[0], 3, self.img_size, slef.img_size))
    return img

class Discriminator(nn.Module):
  def __init__(self, img_size):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(int(np.prod([3, img_size, img_size])), 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, 1),
    )
  def forward(self, img):
    img_flat = torch.reshape(img, (img.shape[0], -1))
    validity = self.model(img_flat)
    return validity

