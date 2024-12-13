#!/usr/bin/python3

from absl import flags, app
import torch
import pytorch_lightning as pl
from merlin.dataloader.torch import Loader
import merlin.models.torch as mm
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag import Ensemble
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.pytorch import PredictPyTorch
from create_datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 10, help = 'epochs')
  flags.DEFINE_string('pipeline', default = 'pipeline', help = 'path to output pipeline')

def main(unused_argv):
  train_transformed, valid_transformed, workflow = load_datasets(FLAGS.dataset, return_workflow = True)
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
    enable_checkpointing = True,
    default_root_dir = FLAGS.ckpt,
    max_epochs = FLAGS.epochs,
    callbacks = [checkpoint_callback],
    log_every_n_steps = 1,
    logger = True
  )
  trainer.lr = FLAGS.lr
  trainer.fit(model, train_dataloaders = Loader(train_transformed, batch_size = FLAGS.batch), val_dataloaders = Loader(valid_transformed, batch_size = FLAGS.batch))
  trainer.validate(model, Loader(valid_transformed, batch_size = FLAGS.batch))
  workflow = workflow.remove_inputs(['binary_rating'])
  pipeline = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> PredictPyTorch(model, train_transformed.schema, Schema([ColumnSchema('binary_rating')]))
  ensemble = Ensemble(pipeline, workflow.input_schema)
  ensemble.export(FLAGS.pipeline)

if __name__ == "__main__":
  add_options()
  app.run(main)
