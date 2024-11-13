#!/usr/bin/python3

from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

def load_datasets(train_val_split = (45000,5000)):
  transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
  ])
  trainval = CIFAR10(root = './cifar10', train = True, download = True, transform = transforms)
  trainset, valset = random_split(trainval, train_val_split)
  return trainset, valset

