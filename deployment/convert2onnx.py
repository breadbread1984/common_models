#!/usr/bin/python3

from absl import flags, app
import onnx
import subprocess

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output', default = 'output.onnx', help = 'path to output onnx model')
  flags.DEFINE_enum('type', default = 'pt', enum_values = {'pt', 'tf'}, help = 'type of input checkpoint type')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device')

def search_command_path(command):
  try:
    result = subprocess.check_output(['which', command]).decode('utf-8').strip()
    return result
  except subprocess.CalledProcessError:
    return None

def main(unused_argv):
  model_path = {
    'pt': 'models/torch_model/1/model.pt',
    'tf': 'models/tensorflow_model/1/model.savedmodel',
  }[FLAGS.type]
  if FLAGS.type == 'pt':
    import torch
    import torchvision
    scripted_model = torch.jit.load(model_path, map_location = FLAGS.device)
    example_input = torch.randn(3,600,800).to(FLAGS.device)
    # NOTE: any problem related to unmatched device, please change https://github.com/pytorch/vision/blob/a9a726a5bde11fc2f1dc2067399e36755b809cb3/torchvision/models/detection/anchor_utils.py#L63 to cuda
    torch.onnx.export(scripted_model, example_input, FLAGS.output,
                      input_names = ['input'],
                      output_names = ['boxes', 'scores', 'labels'],
                      dynamic_axes = {
                        'input': {1: 'height', 2: 'width'},
                        'boxes': {0: 'target_num'},
                        'scores': {0: 'target_num'},
                        'labels': {0: 'target_num'}
                      },
                      opset_version = 13)
  else:
    import tensorflow as tf
    '''
    saved_model = tf.saved_model.load(model_path)
    signature = saved_model.signatures['serving_default']
    input_names = list(signature.structured_input_signature[1].keys())
    output_names = list(signature.structured_outputs.keys())
    '''
    python = search_command_path('python3')
    process = subprocess.Popen([python,
                                "-m",
                                "tf2onnx.convert",
                                "--saved-model",
                                f"{model_path}",
                                "--output",
                                f"{FLAGS.output}",
                                "--opset",
                                "13"])
    try:
      process.wait()
    except KeyboardInterrupt:
      print("Stopping tf2onnx...")
      process.kill()
  onnx_model = onnx.load(FLAGS.output)
  onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
  add_options()
  app.run(main)

