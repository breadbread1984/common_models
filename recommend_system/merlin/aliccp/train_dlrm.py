#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
from merlin.io.dataset import Dataset
import merlin.models.tf as mm
from merlin.schema import ColumnSchema
from merlin.schema.tags import Tags
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'dlrm_ckpt', help = 'path to ckpt')
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 16 * 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 10, help = 'epochs')
  flags.DEFINE_enum('target', default = 'click', enum_values = {'click', 'conversion'}, help = 'which target to use')

def main(unused_argv):
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  valid = Dataset(join(FLAGS.dataset, 'processed', 'valid', '*.parquet'), part_size = "500MB")
  model = mm.DLRMModel(
    train.schema,
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(FLAGS.target),
  )
  model.compile(optimizer="adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
  model.fit(train, validation_data=valid, batch_size=FLAGS.batch)
  model.save(join(FLAGS.ckpt, 'dlrm_model.keras'))

if __name__ == "__main__":
  add_options()
  app.run(main)
