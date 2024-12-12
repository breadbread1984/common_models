#!/usr/bin/python3

from absl import flags, app
import onnx

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output', default = 'output.onnx', help = 'path to output onnx model')
  flags.DEFINE_enum('type', default = 'pt', enum_values = {'pt', 'tf'}, help = 'type of input checkpoint type')

def main(unused_argv):
  model_path = {
    'pt': 'models/torch_model/1/model.pt',
    'tf': 'models/tensorflow_model/1/model.savedmodel',
  }[FLAGS.type]
  if FLAGS.type == 'pt':
    import torch
    scripted_model = torch.jit.load(model_path, map_location = 'cpu')
    example_input = torch.randn(3,600,800)
    torch.onnx.export(scripted_model, example_input, FLAGS.output,
                      input_names = ['input'],
                      output_names = ['boxes', 'scores', 'labels'],
                      dynamic_axes = {
                        'input': {1: 'height', 2: 'width'},
                        'boxes': {0: 'target_num'},
                        'scores': {0: 'target_num'},
                        'labels': {0: 'target_num'}
                      })
  else:
    import tensorflow as tf
    import tf2onnx
    saved_model = tf.saved_model.load(model_path)
    signature = saved_model.signatures['serving_default']
    input_names = list(signature.structured_input_signature[1].keys())
    output_names = list(signature.structured_outputs.keys())
    with tf.device('/cpu:0'):
      onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature = [tf.TensorSpac((None, None, None, 3), tf.float32, name = input_names[0])],
        opset = 13,
        output_path = FLAGS.output
      )
  onnx_model = onnx.load(FLAGS.output)
  onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
  add_options()
  app.run(main)

