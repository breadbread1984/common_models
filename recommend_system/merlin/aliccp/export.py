#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
import torch
import merlin.models.torch as mm
from merlin.schema import ColumnSchema

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = None, help = 'path to ckpt')
  flags.DEFINE_string('output', default = 'aliccp_model', help = 'path to exported model')
  flags.DEFINE_enum('target', default = 'click', enum_values = {'click', 'conversion'}, help = 'which target to use')

def main(unused_argv):
  assert FLAGS.output
  if exists(FLAGS.output): rmtree(FLAGS.output)
  mkdir(FLAGS.output)
  mkdir(join(FLAGS.output,'1'))
  mm.DLRMModel.load_from_checkpoint(FLAGS.ckpt)
  model.eval()
  scripted_model = torch.jit.script(model)
  scripted_model.save(join(FLAGS.output,'1','dlrm_model.pt'))

if __name__ == "__main__":
  add_options()
  app.run(main)

