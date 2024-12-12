#!/usr/bin/python3

from absl import flags, app
import tritonclient.http as httpclient
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = 'localhost', help = 'host')
  flags.DEFINE_integer('port', default = 8081, help = 'port')
  flags.DEFINE_string('input', default = 'pics/YellowLabradorLooking_new.jpg', help = 'path to picture')
  flags.DEFINE_enum('method', default = 'local', enum_values = {'local', 'network'}, help = 'which method to use')
  flags.DEFINE_string('model', default = 'models/tensorflow_model/1/model.savedmodel', help = 'path to model')

def main(unused_argv):
  img = cv2.imread(FLAGS.input)
  inputs = img[:,:,::-1]
  inputs = np.ascontiguousarray(inputs)
  inputs = (inputs / 255).astype(np.float32)
  if FLAGS.method == 'network':
    client = httpclient.InferenceServerClient(f"{FLAGS.host}:{FLAGS.port}")
    feeds = [httpclient.InferInput("inputs", inputs.shape, "FP32")]
    feeds[0].set_data_from_numpy(inputs)
    outputs = [httpclient.InferRequestedOutput('output_0')]
    response = client.infer("tensorflow_model", inputs = feeds, outputs = outputs, model_version = "1")
    features = response.as_numpy("output_0")
  elif FLAGS.method == 'local':
    model = tf.saved_model.load(FLAGS.model)
    features = model(inputs)
  else:
    raise Exception('error method')
  # visualize
  print(features)

if __name__ == "__main__":
  add_options()
  app.run(main)
