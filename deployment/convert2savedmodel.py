#!/usr/bin/python3

from os import makedirs
from os.path import dirname, exists
from absl import flags, app
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('model', enum_values = {'convnext', 'densenet', 'efficientnet', 'inception', 'mobilenet'}, default = 'densenet', help = 'model')
  flags.DEFINE_string('output', default = 'models/tensorflow_model/1/model.savedmodel', help = 'path to saved model')

def main(unused_argv):
  model = {
    'convnext': tf.keras.applications.ConvNeXtLarge,
    'densenet': tf.keras.applications.DenseNet121,
    'efficientnet': tf.keras.applications.EfficientNetV2L,
    'inception': tf.keras.applications.InceptionV3,
    'mobilenet': tf.keras.applications.MobileNetV3Large
  }[FLAGS.model](weights = 'imagenet', include_top = False)
  tf.saved_model.save(model, FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)

