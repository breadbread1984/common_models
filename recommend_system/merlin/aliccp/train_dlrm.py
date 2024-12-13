#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
from merlin.io.dataset import Dataset
import merlin.models.tf as mm
from merlin.schema import ColumnSchema
from merlin.schema.tags import Tags
from merlin.systems.dag import Ensemble
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'dlrm_ckpt', help = 'path to ckpt')
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 16 * 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 2, help = 'epochs')
  flags.DEFINE_enum('target', default = 'click', enum_values = {'click', 'conversion'}, help = 'which target to use')
  flags.DEFINE_string('pipeline', default = 'pipeline_tf', help = 'path to output pipeline')

def main(unused_argv):
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  valid = Dataset(join(FLAGS.dataset, 'processed', 'valid', '*.parquet'), part_size = "500MB")
  model = mm.DLRMModel(
    train.schema.without(['click', 'conversion']),
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(FLAGS.target),
  )
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
  model.compile(optimizer = optimizer, run_eagerly = False, metrics=[tf.keras.metrics.AUC()])
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = join(FLAGS.ckpt, 'logs')),
    #tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.ckpt, 'dlrm_ckpt'), save_weights_only = True, save_freq = 'epoch')
  ]
  model.fit(train, validation_data=valid, batch_size=FLAGS.batch, epochs = FLAGS.epochs, callbacks = callbacks)
  metrics = model.evaluate(valid, batch_size = FLAGS.batch, return_dict = True)
  print(metrics)
  # export to pipeline
  workflow = nvt.Workflow.load('dlrm_torch.workflow')
  workflow.remove_inputs(['click', 'conversion'])
  pipeline = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> PredictTensorflow(model)
  ensemble = Ensemble(pipeline, workflow.input_schema)
  ensemble.export(FLAGS.pipeline)

if __name__ == "__main__":
  add_options()
  app.run(main)
