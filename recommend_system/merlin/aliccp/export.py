#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
import torch
import merlin.models.torch as mm
from merlin.schema import ColumnSchema
from merlin.io.dataset import Dataset
from merlin.dataloader.torch import Loader

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
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  loader = Loader(train, batch_size = 1)
  model = mm.DLRMModel(
    train.schema,
    dim = 64,
    bottom_block = mm.MLPBlock([128, 64]),
    top_block = mm.MLPBlock([128, 64, 32]),
    output_block = mm.BinaryOutput(ColumnSchema(FLAGS.target)),
  )
  model.eval()
  device = next(model.parameters()).device
  model({k:v.to(device) for k,v in next(loader)[0].items()})
  model.load_state_dict(torch.load(FLAGS.ckpt)['state_dict'])
  scripted_model = torch.jit.script(model)
  scripted_model.save(join(FLAGS.output,'1','dlrm_model.pt'))

if __name__ == "__main__":
  add_options()
  app.run(main)

