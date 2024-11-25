#!/usr/bin/python3

from shutil import rmtree
from os.path import join, exists
from absl import flags, app
from merlin.datasets.ecommerce import get_aliccp
from merlin.core.dispatch import get_lib
from merlin.schema.tags import Tags

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'output directory')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  get_aliccp(path = FLAGS.output_dir)

def load_datasets(root_path = 'dataset'):
  train = get_lib().read_parquet(join(root_path, 'transformed', 'train.parquet'))
  valid = get_lib().read_parquet(join(root_path, 'transformed', 'valid.parquet'))
  user_id_raw = []

if __name__ == "__main__":
  add_options()
  app.run(main)

