#!/usr/bin/python3

from absl import flags, app
from torchtext.datasets import IMDB
from datasets import Dataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output', default = 'imdb', help = 'path to output dataset')

def main(unused_argv):
  train, test = IMDB()
  for label, text in train:
    print(label, text)
    break

if __name__ == "__main__":
  add_options()
  app.run(main)
