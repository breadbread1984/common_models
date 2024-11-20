#!/usr/bin/python3

import numpy as np
import torch
from torch import nn

class Generator(nn.Module):
  def __init__(self, latent_dim = 100):
    super(Generator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(latent_dim, 256 * 4 * 4),
      nn.Unflatten(1, (256,4,4)),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace = True),
      nn.ConvTranspose2d(256, 128, kernel_size = (4,4), stride = (2,2), padding = 'same'), # shape = (batch, 128, 8, 8)
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace = True),
      nn.ConvTranspose2d(128, 64, kernel_size = (4,4), stride = (2,2), padding = 'same'), # shape = (batch, 64, 16, 16)
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2, inplace = True),
      nn.ConvTranspose2d(64, 3, kernel_size = (4,4), stride = (2,2), padding = 'same'), # shape = (batch, 3, 32, 32)
      nn.Tanh()
    )
  def forward(self, z):
    img = self.model(z)
    return img

class Discriminator(nn.Module):
  def __init__(self, ):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size = (4,4), stride = (2,2), padding = 'same'), # shape = (batch, 64, 16, 16)
      nn.LeakyReLU(0.2, inplace = True),
      nn.Dropout(0.3),
      nn.Conv2d(64, 128, kernel_size = (4,4), stride = (2,2), padding = 'same'), # shape = (batch, 128, 8, 8)
      nn.LeakyReLU(0.2, inplace = True),
      nn.Dropout(0.3),
      nn.Conv2d(128, 256, kernel_size = (4,4), stride = (2,2), padding = 'same'), # shape = (batch, 256, 4, 4)
      nn.LeakyReLU(0.2, inplace = True),
      nn.Dropout(0.3),
      nn.Flatten(),
      nn.Linear(256 * 4 * 4, 1),
      nn.Sigmoid()
    )
  def forward(self, img):
    validity = self.model(img)
    return validity

