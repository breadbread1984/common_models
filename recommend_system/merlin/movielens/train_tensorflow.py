#!/usr/bin/python3

from os.path import join, exists
from absl import flags, app
import tensorflow as tf
import merlin.models.tf as mm
from merlin.systems.dag import Ensemble
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from create_datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 10, help = 'epochs')
  flags.DEFINE_string('pipeline', default = 'pipeline_tf', help = 'path to output pipeline')

def main(unused_argv):
  train_transformed, valid_transformed, workflow = load_datasets(FLAGS.dataset, return_workflow = True)
  # 1)stack categorical input vectors with continuous input vectors to form a matrix F of n x dim(64)
  # 2)interaction is done by dot(F, transpose(F)) whose dimension is n x n
  # 3)all elements in the upper triangle are flatten in to a vector
  # 4)map feature through top_block to map the interaction to target vector
  model = mm.DLRMModel(
    train_transformed.schema.without(['binary_rating']), # .without('genres')
    embedding_dim = 64, # embedding dim for categorical inputs
    bottom_block = mm.MLPBlock([128, 64]), # mlp for continous inputs (no continous input for this case)
    top_block = mm.MLPBlock([128, 64, 32]), # mlp after interaction block
    prediction_tasks = mm.OutputBlock(train_transformed.schema)
  )
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
  model.compile(optimizer = optimizer)
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = join(FLAGS.ckpt, 'logs')),
    # NOTE: https://github.com/NVIDIA-Merlin/models/blob/eb1e54196a64a70950b2a7e7744d2150e052d53e/merlin/models/tf/models/base.py#L1687
    # merlin.models.tf.models.Model's save function doesn't support overwrite argument
    #tf.keras.callbacks.ModelCheckpoint(filepath = FLAGS.ckpt, save_freq = 'epoch')
  ]
  model.fit(train_transformed, epochs = FLAGS.epochs, validation_data = valid_transformed, batch_size = FLAGS.batch, callbacks = callbacks)
  model.evaluate(valid_transformed, batch_size = FLAGS.batch)
  workflow = workflow.remove_inputs(['binary_rating'])
  pipeline = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> PredictTensorflow(model)
  ensemble = Ensemble(pipeline, workflow.input_schema)
  ensemble.export(FLAGS.pipeline)

if __name__ == "__main__":
  add_options()
  app.run(main)
