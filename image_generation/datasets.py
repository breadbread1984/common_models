#!/usr/bin/python3

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

def load_datasets():
  transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
  ])
  trainset = CIFAR10(root = './cifar10', train = True, download = True, transform = transforms)
  return trainset

