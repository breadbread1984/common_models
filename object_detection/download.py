#!/usr/bin/python3

from absl import flags, app
from abc import ABC, abstractmethod
from shutil import rmtree
from os import mkdir
from os.path import join, exists
import gdown
import zipfile
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
    if not exists(target_path):
      mkdir(target_path)
      with zipfile.ZipFile(download_path, 'r') as f:
        f.extractall(target_path)


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
    if not exists(download_file)
    gdown.download(data_info['url'], output = download_file)
    processor = data_info['postprocess']()
    processor.process(download_file)
  else:
    raise Exception('unknown type of dataset')

if __name__ == "__main__":
  add_options()
  app.run(main)

