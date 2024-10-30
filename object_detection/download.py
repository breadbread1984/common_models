#!/usr/bin/python3

from absl import flags, app
from abc import ABC, abstractmethod
from shutil import rmtree
from os import mkdir, listdir
from os.path import join, exists, splitext
import gdown
import zipfile
import csv
from utils import voc_to_coco

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', enum_values = {'safetyhelmet'}, default = 'safetyhelmet', help = 'name of dataset')
  flags.DEFINE_string('output', default = 'datasets', help = 'dataset download path')

class PostProcess(ABC):
  @abstractmethod
  def process(self, download_path):
    raise NotImplementedError('PostProcess.process is not implemented!')

class SafetyHelmet(PostProcess):
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
      with open(join(target_path, 'VOC2028', 'ImageSets', 'Main', list_file)) as f:
        sample_list = set()
        reader = csv.reader(f)
        for row in reader:
          if len(row) == 0: continue
          sample_list.add(row[0])
      images = list(filter(lambda x: x["id"] in sample_list, coco_ann['images']))
      image_ids = set([image['id'] for image in images])
      annotations = list(filter(lambda x: x["image_id"] in image_ids, coco_ann['annotations']))
      subset_ann = {
        "images": images,
        "annotations": annotations,
        "categories": coco_ann['categories']
      }
      with open(join(label_path, stem + '.json'), 'w') as f:
        f.write(json.dumps(subset_ann))

def main(unused_argv):
  if not exists(FLAGS.output):
    mkdir(FLAGS.output)
  datasets = {
    'safetyhelmet': {
      'type': 'google drive',
      'url': 'https://drive.google.com/uc?id=1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX',
      'postprocess': SafetyHelmet,
    },
  }
  data_info = datasets[FLAGS.dataset]
  download_file = join(FLAGS.output, FLAGS.dataset + '.download')
  if data_info['type'] == 'google drive':
    if not exists(download_file):
      gdown.download(data_info['url'], output = download_file)
    processor = data_info['postprocess']()
    processor.process(download_file)
  else:
    raise Exception('unknown type of dataset')

if __name__ == "__main__":
  add_options()
  app.run(main)

