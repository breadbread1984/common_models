#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
from merlin.io.dataset import Dataset
import merlin.models.torch as mm
from merlin.schema import ColumnSchema
from merlin.schema.tags import Tags
import torch
import pytorch_lightning as pl
from merlin.dataloader.torch import Loader

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 10, help = 'epochs')
  flags.DEFINE_enum('target', default = 'click', enum_values = {'click', 'conversion'}, help = 'which target to use')

def main(unused_argv):
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  valid = Dataset(join(FLAGS.dataset, 'processed', 'valid', '*.parquet'), part_size = "500MB")
  # train.schema.select_by_tag(Tags.TARGET).column_names
  model = mm.DLRMModel(
    train.schema,
    dim = 64,
    bottom_block = mm.MLPBlock([128, 64]),
    top_block = mm.MLPBlock([128, 64, 32]),
    output_block = mm.BinaryOutput(ColumnSchema(FLAGS.target)),
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

