#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists
import gdown
from datasets import iSAID

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', enum_values = {'isaid'}, default = 'isaid', help = 'name of dataset')
  flags.DEFINE_string('output', default = 'datasets', help = 'dataset download path')

def main(unused_argv):
  if not exists(FLAGS.output):
    mkdir(FLAGS.output)
  datasets = {
    'isaid': {
      'type': 'google drive',
      'url': {
        'train_image': 'https://drive.google.com/drive/folders/1MvSH7sNaY4p4lhwAU_BG3y7zth6-rtrD',
        'val_image': 'https://drive.google.com/drive/folders/1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ',
        'train': 'https://drive.google.com/drive/folders/19RPVhC0dWpLF9Y_DYjxjUrwLbKUBQZ2K',
        'val': 'https://drive.google.com/drive/folders/17MErPhWQrwr92Ca1Maf4mwiarPS5rcWM'
      },
      'postprocess': iSAID,
    }
  }
  data_info = datasets[FLAGS.dataset]
  download_path = join(FLAGS.output, FLAGS.dataset + '.download')
  if data_info['type'] == 'google drive':
    if not exists(download_path):
      if type(data_info['url']) is str:
        if data_info['url'].find('folder') >= 0:
          gdown.download_folder(data_info['type'], output = download_path)
        else:
          gdown.download(data_info['type'], output = download_path)
      elif type(data_info['url']) is dict:
        if not exists(download_path): mkdir(download_path)
        for split, url in data_info['url'].items():
          if url.find('folder') >= 0:
            gdown.download_folder(url, output = join(download_path, split))
          else:
            gdown.download(url, output = join(download_path, split))
      else:
        raise Exception('unknown type of url')
  else:
    raise Exception('unknown type of dataset')
  processor = data_info['postprocess']()
  processor.process(download_path, output = FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)

