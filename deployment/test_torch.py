#!/usr/bin/python3

from absl import flags, app
import requests
import numpy as np
import cv2
import torch

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = "localhost", help = 'host')
  flags.DEFINE_integer('port', default = 8081, help = 'port')
  flags.DEFINE_string('input', default = 'pics/000035.jpg', help = 'path to picture')
  flags.DEFINE_enum('method', default = 'local', enum_values = {'local', 'network'}, help = 'which method to use')
  flags.DEFINE_string('model', default = 'converted.pt', help = 'path to model')

def main(unused_argv):
  img = cv2.imread(FLAGS.input)
  img = img[:,:,::-1]
  img = np.ascontiguousarray(img)
  img = (img / 255).astype(np.float32)
  img = np.transpose(img, (2,0,1))
  inputs = np.expand_dims(img, axis = 0)
  if FLAGS.method == 'network':
    data = {"inputs": {'%inputs': inputs.tolist()}}
    response = requests.post(
      f"http://{FLAGS.host}:{FLAGS.port}",
      headers = {"Content-Type": "application/json"},
      json = data
    )
    assert response.status_code == 200
    res = response.json()
  elif FLAGS.method == 'local':
    model = torch.jit.load(FLAGS.model, map_location = 'cuda')
    output = model.forward([torch.from_numpy(inputs)])
  else:
    raise Exception('error method')
  import pdb; pdb.set_trace()

if __name__ == "__main__":
  add_options()
  app.run(main)

