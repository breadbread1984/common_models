#!/usr/bin/python3

from absl import flags, app
import tensorflow
import merline.models.tf as mm

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
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.RegressionTask('rating')
 )

  opt = tensorflow.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer=opt)
  model.fit(train_transformed, validation_data=valid_transformed, batch_size=FLAGS.batch, epochs=FLAGS.epochs)

  model.optimizer.learning_rate = FLAGS.lr
  metrics = model.fit(train_transformed, validation_data=valid_transformed, batch_size=FLAGS.batch, epochs=3)

if __name__ == "__main__":
  add_options()
  app.run(main)
