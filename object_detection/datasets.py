#!/usr/bin/python3

from os.path import join, exists
from abc import ABC, abstractmethod
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

class CocoDataset(ABC):
  @abstractmethod
  def load(self, root_path):
    raise NotImplementedError('CocoDataset.load is not implemented!')

class SafetyHelmet(CocoDataset):
  def load(self, root_path, split = 'train', transforms = None):
    assert split in {'train', 'val'}
    PATHS = {
      "train": (join(root_path, 'VOC2028', 'JPEGImages'), join(root_path, 'labels', 'train.json')),
      "val": (join(root_path, 'VOC2028', 'JPEGImages'), join(root_path, 'labels', 'val.json'))
    }
    img_folder, ann_file = PATHS[split]
    dataset = CocoDetection(img_folder, ann_file, transforms = transforms)
    target_keys = ['boxes', 'labels', 'image_id']
    dataset = wrap_dataset_for_transforms_v2(dataset, target_keys = target_keys)
    if split == 'train':
      ids = list()
      for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds = img_id, iscrowd = None)
        anno = dataset.coco.loadAnns(ann_ids)
        if not len(anno) == 0 and \
           not all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno):
          ids.append(ds_idx)
      dataset = torch.utils.data.Subset(dataset, ids)
    return dataset

if __name__ == "__main__":
  trainset = SafetyHelmet().load('datasets/safetyhelmet','train')
  testset = SafetyHelmet().load('datasets/safetyhelmet','val')
