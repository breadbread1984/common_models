#!/usr/bin/python3

from absl import flags, app
import subprocess

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to onnx')
  flags.DEFINE_string('output', default = 'output.engine', help = 'path to tensorrt engine')
  flags.DEFINE_enum('type', default = 'pt', enum_values = {'pt', 'tf'}, help = 'type')

def search_command_path(command):
  try:
    result = subprocess.check_output(['which', command]).decode('utf-8').strip()
    return result
  except subprocess.CalledProcessError:
    return None

def main(unused_argv):
  trtexec = search_command_path('trtexec')
  assert trtexec is not None
  if FLAGS.type == 'pt':
    process = subprocess.Popen([
      trtexec,
      f"--onnx={FLAGS.input}",
      "--minShapes=input:3x224x224",
      "--maxShapes=input:3x1000x1000",
      "--optShapes=input:3x224x224",
      f"--saveEngine={FLAGS.output}",
      "--fp16"
    ])
  else:
    process = subprocess.Popen([
      trtexec,
      f"--onnx={FLAGS.input}",
      "--minShapes=keras_input:1x224x224x3",
      "--maxShapes=keras_input:32x1000x1000x3",
      "--optShapes=keras_input:1x224x224x3",
      f"--saveEngine={FLAGS.output}",
      "--fp16"
    ])

if __name__ == "__main__":
  add_options()
  app.run(main)

