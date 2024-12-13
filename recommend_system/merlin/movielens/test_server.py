#!/usr/bin/python3

from absl import flags, app
import tritonclient.http as httpclient
from create_datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = 'localhost', help = 'host')
  flags.DEFINE_integer('port', default = 8081, help = 'port')
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset')

def main(unused_argv):
  _, valid = load_datasets(FLAGS.dataset)
  client = httpclient.InferenceServerClient(url = f"{FLAGS.host}:{FLAGS.port}")
  if not client.is_model_ready('executor_model'):
    raise Exception('Model is not ready!')
  df = valid.to_ddf():
  for index, row in df.iterrows():
    import pdb; pdb.set_trace()

if __name__ == "__main__":
  add_options()
  app.run(main)

