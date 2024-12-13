#!/usr/bin/python3

from absl import flags, app
import torch
import pytorch_lightning as pl
from merlin.dataloader.torch import Loader
import merlin.models.torch as mm
from merlin.schema import ColumnSchema
from create_datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  train_transformed, valid_transformed = load_datasets(FLAGS.dataset)
  model = mm.DLRMModel(
    train_transformed.schema, # .without('genres')
    dim = 64, # embedding dim for categorical inputs
    bottom_block = mm.MLPBlock([128, 64]), # mlp for continous inputs (no continous input for this case)
    top_block = mm.MLPBlock([128, 64, 32]), # mlp after interaction block
    output_block = mm.BinaryOutput(ColumnSchema('binary_rating'))
  )
  trainer = pl.Trainer(
    enable_checkpointing = False,
    default_root_dir = FLAGS.ckpt,
  )
  trainer.validate(model, Loader(valid_transformed, batch_size = FLAGS.batch), ckpt_path = 'last')
  model.to(FLAGS.device)
  script_model = torch.jit.script(model)
  torch.jit.save(script_model, 'model.pt')

if __name__ == "__main__":
  add_options()
  app.run(main)

