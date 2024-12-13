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
from merlin.systems.dag import Ensemble
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.pytorch import PredictPyTorch
import nvtabular as nvt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'dlrm_ckpt', help = 'path to ckpt')
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 10, help = 'epochs')
  flags.DEFINE_enum('target', default = 'click', enum_values = {'click', 'conversion'}, help = 'which target to use')
  flags.DEFINE_string('pipeline', default = 'pipeline', help = 'path to exported pipeline')

def main(unused_argv):
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  valid = Dataset(join(FLAGS.dataset, 'processed', 'valid', '*.parquet'), part_size = "500MB")
  # train.schema.select_by_tag(Tags.TARGET).column_names
  model = mm.DLRMModel(
    train.schema.without(['click', 'conversion']),
    dim = 64,
    bottom_block = mm.MLPBlock([128, 64]),
    top_block = mm.MLPBlock([128, 64, 32]),
    output_block = mm.BinaryOutput(ColumnSchema(FLAGS.target)),
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
  trainer.fit(model, train_dataloaders = Loader(train, batch_size = FLAGS.batch), val_dataloaders = Loader(valid, batch_size = FLAGS.batch))
  trainer.validate(model, Loader(valid, batch_size = FLAGS.batch), ckpt_path = 'last')
  # export to pipeline
  workflow = nvt.Workflow.load('dlrm_torch.workflow')
  workflow.remove_inputs(['click', 'conversion'])
  pipeline = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> PredictPyTorch(model, model.input_schema.without(['click','conversion']), model.output_schema)
  ensemble = Ensemble(pipeline, workflow.input_schema)
  ensemble.export(FLAGS.pipeline)

if __name__ == "__main__":
  add_options()
  app.run(main)

