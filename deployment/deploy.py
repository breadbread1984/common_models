#!/usr/bin/python3

from absl import flags, app
import tritonserver

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('model', default = 'converted.trt', help = 'path to model')
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'host')
  flags.DEFINE_integer('port', default = 8081, help = 'port')

def main(unused_argv):
  triton_server = tritonserver.InferenceServerREST(f'{FLAGS.host}:{FLAGS.port}')
  triton_server.load_model(FLAGS.model)

if __name__ == "__main__":
  add_options()
  app.run(main)

