#!/usr/bin/python3

from absl import flags, app
from transformers import BertForTokenClassification

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')

def main(unused_argv):
  model = BertForTokenClassification.from_pretrained('google-bert/bert-base-uncased', num_labels = 2)

if __name__ == "__main__":
  add_options()
  app.run(main)

