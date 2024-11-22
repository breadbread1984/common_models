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
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 5, help = 'epochs')

def main(unused_argv):
  train_transformed, valid_transformed = load_datasets(FLAGS.dataset)
  model = mm.DLRMModel(
    train_transformed.schema,
    dim = 64,
    bottom_block = mm.MLPBlock([128, 64]), # mlp for input
    top_block = mm.MLPBlock([128, 64, 32]), # mlp for output
    output_block = mm.RegressionOutput(ColumnSchema('rating'))
  )
  trainer = pl.Trainer(
    enable_checkpointing = True,
    default_root_dir = FLAGS.ckpt,
    max_epochs = FLAGS.epochs)
  trainer.lr = FLAGS.lr
  trainer.fit(model, train_dataloaders = Loader(train_transformed, batch_size = FLAGS.batch), val_dataloaders = Loader(valid_transformed, batch_size = FLAGS.batch))
  trainer.validate(model, Loader(valid_transformed, batch_size = FLAGS.batch))

if __name__ == "__main__":
  add_options()
  app.run(main)
