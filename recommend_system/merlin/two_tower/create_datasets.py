#!/usr/bin/python3

from os.path import join, exists
import tarfile
import pandas as pd
from merlin.datasets.synthetic import generate_data
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema.tags import Tags

def load_datasets(train_file_path):
  '''
  if not exists('train'):
    with tarfile.open(train_file_path, 'r:gz') as tar:
      tar.extractall(path = 'train')
  sample_sekeleton = pd.read_csv(
    join('train','sample_skeleton_train.csv'),
    header = None,
    names = ["sampleId", "clickLabel", "conversionLabel", "commonFeatureIndex", "featureNum", "featureList"])
  common_features = pd.read_csv(
    join('train','common_features_train.csv'),
    header = None,
    names = ['commonFeatureIndex', 'featureNum', 'featureList'])
  '''
  train_raw, valid_raw = generate_data('aliccp-raw', 100000, set_sizes = (0.7, 0.3))
  user_features = unique_rows_by_features(train_raw, Tags.USER, Tags.USER_ID).compute().reset_index(drop=True)
  print(user_features.head())

if __name__ == "__main__":
  load_datasets(None)
