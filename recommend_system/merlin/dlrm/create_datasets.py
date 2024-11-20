#!/usr/bin/python3

from shutil import rmtree
from os.path import join, exists
from absl import flags, app
from merlin.datasets.entertainment import get_movielens
from merlin.core.dispath import get_lib
from merlin.schema.tags import Tags
from nvtabular as nvt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'output directory')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  get_movielens(variant="ml-1m", path = FLAGS.output_dir)

def load_datasets(root_path):
  train = get_lib().read_parquet(join(root_path, 'ml-1m', 'train.parquet'))
  valid = get_lib().read_parquet(join(root_path, 'ml-1m', 'valid.parquet'))
  train_ds = nvt.Dataset(train)
  valid_ds = nvt.Dataset(valid)
  return train_ds, valid_ds

if __name__ == "__main__":
  add_options()
  app.run(main)
