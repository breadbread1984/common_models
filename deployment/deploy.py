#!/usr/bin/python3

from absl import flags, app
import subprocess

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('model', default = 'torch_model_repo', help = 'path to model')
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'host')
  flags.DEFINE_integer('port', default = 8081, help = 'port')

def search_command_path(command):
  try:
    result = subprocess.check_output(['which', command]).decode('utf-8').strip()
    return result
  except subprocess.CalledProcessError:
    return None

def main(unused_argv):
  full_path = search_command_path('tritonserver')
  assert full_path is not None
  process = subprocess.Popen([full_path,
                              f"--model-repository={FLAGS.model}",
                              f"--http-port={FLAGS.port}",
                              f"--grpc-port={FLAGS.port+1}",
                              f"--metrics-port={FLAGS.port+2}"])
  try:
    process.wait()
  except KeyboardInterrupt:
    print("Stopping Triton Server...")
    process.kill()

if __name__ == "__main__":
  add_options()
  app.run(main)

