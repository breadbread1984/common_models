#!/usr/bin/python3

from abc import ABC, abstractmethod
from os.path import join, exists
import zipfile
import gdown
import torch
from torchvision.datasets import CocoDetection
from transforms import Compose
from coco_utils import FilterAndRemapCocoCategories, ConvertCocoPolysToMask

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
    with zipfile.ZipFile(join(download_path, 'train_image', 'part1.zip'), 'r') as f:
      f.extractall(join(download_path, 'train_image'))
    with zipfile.ZipFile(join(download_path, 'train_image', 'part2.zip'), 'r') as f:
      f.extractall(join(download_path, 'train_image'))
    with zipfile.ZipFile(join(download_path, 'train_image', 'part3.zip'), 'r') as f:
      f.extractall(join(download_path, 'train_image'))
    with zipfile.ZipFile(join(download_path, 'val_image', 'part1.zip'), 'r') as f:
      f.extractall(join(download_path, 'val_image'))
    with zipfile.ZipFile(join(download_path, 'train', 'Instance_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'train', 'Instance_masks'))
    with zipfile.ZipFile(join(download_path, 'train', 'Semantic_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'train', 'Semantic_masks'))
    with zipfile.ZipFile(join(download_path, 'val', 'Instance_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'val', 'Instance_masks'))
    with zipfile.ZipFile(join(download_path, 'val', 'Semantic_masks', 'images.zip'), 'r') as f:
      f.extractall(join(download_path, 'val', 'Semantic_masks'))
  def load(self, root_path, split = 'train', transforms = None):
    assert split in {'train', 'val'}
    PATHS = {
      "train": (join(root_path, "train_image", "images"), join(root_path, 'train', 'Annotations', 'iSAID_train.json')),
      "val": (join(root_path, "val_image", "images"), join(root_path, "val", "Annotations", "iSAID_val.json"))
    }
    img_folder, ann_file = PATHS[split]
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask(), transforms])
    dataset = CocoDetection(img_folder, ann_file, transforms = transforms)
    if split == 'train':
      ids = list()
      for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds = img_id, iscrowd = None)
        anno = dataset.coco.loadAnns(ann_ids)
        if not len(anno) == 0 and \
           sum(obj["area"] for obj in anno) > 1000:
          ids.append(ds_idx)
      dataset = torch.utils.data.Subset(dataset, ids)
    return dataset

if __name__ == "__main__":
  from presets import SegmentationPresetEval
  isaid = iSAID()
  transforms = SegmentationPresetEval(base_size = 520, backend = 'PIL', use_v2 = False)
  valset = isaid.load(join('datasets', 'isaid.download'), split = 'val', transforms = transforms)
  for sample in valset:
    import pdb; pdb.set_trace()
