#!/usr/bin/python3

from datasets import load_dataset
from torchvision.transforms import transforms

def load_datasets(config):
  config.dataset_name = "huggan/smithsonian_butterflies_subset"
  dataset = load_dataset(config.dataset_name, split = "train")
  trans = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
  ])
  dastaset.set_transform(lambda examples: {"images": [trans(image.convert('RGB')) for image in examples["image"]]})
  return dataset

