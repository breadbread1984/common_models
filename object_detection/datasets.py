#!/usr/bin/python3

from os.path import join, exists
from abc import ABC, abstractmethod
from os import mkdir, listdir
from os.path import join, exists, splitext
import gdown
import zipfile
import csv
import json
import torch
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

class CocoDataset(ABC):
  @abstractmethod
  def process(self, download_path):
    raise NotImplementedError('CocoDataset.process is not implemented!')
  @abstractmethod
  def load(self, root_path):
    raise NotImplementedError('CocoDataset.load is not implemented!')

class SafetyHelmet(CocoDataset):
  def process(self, download_path):
    target_path = join(FLAGS.output, 'safetyhelmet')
    # 1) unzip dataset
    if not exists(target_path):
      mkdir(target_path)
      with zipfile.ZipFile(download_path, 'r') as f:
        f.extractall(target_path)
    # 2) convert VOC to COCO annotation
    coco_ann = voc_to_coco(join(target_path, 'VOC2028', 'Annotations'))
    # 3) load list
    label_path = join(FLAGS.output, 'safetyhelmet', 'labels')
    if not exists(label_path):
      mkdir(label_path)
    for list_file in listdir(join(target_path, 'VOC2028', 'ImageSets', 'Main')):
      stem, ext = splitext(list_file)
      if ext != '.txt': continue
      with open(join(target_path, 'VOC2028', 'ImageSets', 'Main', list_file), 'r') as f:
        sample_list = set()
        reader = csv.reader(f)
        for row in reader:
          if len(row) == 0: continue
          sample_list.add(row[0])
      images = list(filter(lambda x: splitext(x["file_name"])[0] in sample_list, coco_ann['images']))
      image_ids = set([image['id'] for image in images])
      annotations = list(filter(lambda x: x["image_id"] in image_ids, coco_ann['annotations']))
      subset_ann = {
        "images": images,
        "annotations": annotations,
        "categories": coco_ann['categories']
      }
      with open(join(label_path, stem + '.json'), 'w') as f:
        f.write(json.dumps(subset_ann, indent = 2, ensure_ascii = False))
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
  safetyhelmet = SafetyHelmet()
  trainset = safetyhelmet.load('datasets/safetyhelmet','train')
  testset = safetyhelmet.load('datasets/safetyhelmet','val')
