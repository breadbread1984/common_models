#!/usr/bin/python3

from absl import flags, app
import tritonhttpclient
import tritonclient.grpc as grpcclient
import merlin.systems.triton as merlin_triton

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'host')
  flags.DEFINE_integer('port', default = 8000, help = 'port')

def main(unused_argv):
  triton_client = tritonhttpclient.InferenceServerClient(url = f"{FLAGS.host}:{FLAGS.port}", verbose = True)
  assert triton_client.is_server_live()
  triton_client.load_model(FLAGS.ckpt)
