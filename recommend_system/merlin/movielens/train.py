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
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 10, help = 'epochs')
  flags.DEFINE_boolean('eval_only', default = False, help = 'whether to do just evaluation')

def main(unused_argv):
  train_transformed, valid_transformed = load_datasets(FLAGS.dataset)
  # 1)stack categorical input vectors with continuous input vectors to form a matrix F of n x dim(64)
  # 2)interaction is done by dot(F, transpose(F)) whose dimension is n x n
  # 3)all elements in the upper triangle are flatten in to a vector
  # 4)map feature through top_block to map the interaction to target vector
  model = mm.DLRMModel(
    train_transformed.schema, # .without('genres')
    dim = 64, # embedding dim for categorical inputs
    bottom_block = mm.MLPBlock([128, 64]), # mlp for continous inputs (no continous input for this case)
    top_block = mm.MLPBlock([128, 64, 32]), # mlp after interaction block
    output_block = mm.BinaryOutput(ColumnSchema('binary_rating'))
  )
  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = 'val_binary_accuracy',
    mode = 'max',
    save_top_k = 2,
    save_last = True
  )
  trainer = pl.Trainer(
    enable_checkpointing = True if not FLAGS.eval_only else False,
    default_root_dir = FLAGS.ckpt,
    max_epochs = FLAGS.epochs,
    callbacks = [checkpoint_callback] if not FLAGS.eval_only else [],
    log_every_n_steps = 1,
    logger = True
  )
  trainer.lr = FLAGS.lr
  if not FLAGS.eval_only:
    trainer.fit(model, train_dataloaders = Loader(train_transformed, batch_size = FLAGS.batch), val_dataloaders = Loader(valid_transformed, batch_size = FLAGS.batch))
    trainer.validate(model, Loader(valid_transformed, batch_size = FLAGS.batch))
  else:
    trainer.validate(model, Loader(valid_transformed, batch_size = FLAGS.batch), ckpt_path = 'last')

if __name__ == "__main__":
  add_options()
  app.run(main)
