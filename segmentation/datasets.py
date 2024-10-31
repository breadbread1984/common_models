#!/usr/bin/python3

from abc import ABC, abstractmethod
from os.path import join, exists
import zipfile
import gdown

class SegmentDataset(ABC):
  @abstractmethod
  def process(self, download_path):
    raise NotImplementedError('SegmentDataset.process is not implemented!')
  @abstractmethod
  def load(self, root_path):
    raise NotImplementedError('SegmentDataset.load is not implemented!')

class iSAID(SegmentDataset):
  def process(self, download_path, output = 'datasets'):
    target_path = join(output, 'isaid')
    # 1) unzip dataset
    with zipfile.ZipFile(join(download_path, 'train', 'Instance_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'train', 'Instance_masks'))
    with zipfile.ZipFile(join(download_path, 'train', 'Semantic_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'train', 'Semantic_masks'))
    with zipfile.ZipFile(join(download_path, 'val', 'Instance_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'val', 'Instance_masks'))
    with zipfile.ZipFile(join(download_path, 'val', 'Semantic_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'val', 'Semantic_masks'))
    # 2)
  def load(self, root_path):
    pass
