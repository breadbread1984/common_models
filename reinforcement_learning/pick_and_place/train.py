#!/usr/bin/python3

from absl import flags, app
from create_datasets import load_fetchpickplace_env

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')

def main(unused_argv):
  pass

if __name__ == "__main__":
  add_options()
  app.run(main)

