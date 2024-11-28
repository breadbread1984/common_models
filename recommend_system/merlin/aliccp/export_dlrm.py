#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
import tensorflow as tf
import merlin.models.tf as mm
from merlin.schema import ColumnSchema
from merlin.io.dataset import Dataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = None, help = 'path to ckpt')
  flags.DEFINE_integer('batch', default = 16 * 1024, help = 'batch size')
  flags.DEFINE_string('output', default = 'dlrm_model', help = 'path to exported model')
  flags.DEFINE_enum('target', default = 'click', enum_values = {'click', 'conversion'}, help = 'which target to use')

def main(unused_argv):
  assert FLAGS.output
  if exists(FLAGS.output): rmtree(FLAGS.output)
  mkdir(FLAGS.output)
  mkdir(join(FLAGS.output,'1'))
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  model = mm.DLRMModel(
    train.schema,
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(FLAGS.target),
  )
  model.compile(optimizer="adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
  model.fit(train, batch_size = FLAGS.batch, epochs = 1, steps_per_epoch = 1)
  model.load_weights(join(FLAGS.ckpt, 'dlrm_ckpt'))
  tf.saved_model.save(model, join(FLAGS.output, '1'))

if __name__ == "__main__":
  add_options()
  app.run(main)

