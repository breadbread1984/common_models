#!/usr/bin/python3

from shutil import rmtree
from os.path import join, exists
from absl import flags, app
from merlin.datasets.entertainment import get_movielens

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'output directory')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  get_movielens(variant="ml-1m", path = FLAGS.output_dir)

if __name__ == "__main__":
  add_options()
  app.run(main)
