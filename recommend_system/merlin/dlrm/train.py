#!/usr/bin/python3

from absl import flags, app
import torch
import merlin.models.torch as mm
from create_datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset')

def main(unused_argv):
  train_transformed, valid_transformed = load_datasets(FLAGS.dataset)
  model = mm.DLRMModel(
    train_transformed.schema,
    embedding_dim = 64,
    bottom_block = mm.MLPBlock([128, 64]),
    top_block = mm.MLPBlock([128, 64, 32]),
    prediction_tasks = mm.RegressionTask('rating')
  )
