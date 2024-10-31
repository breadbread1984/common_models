#!/usr/bin/python3

from abc import ABC, abstractmethod
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
    # 1)
