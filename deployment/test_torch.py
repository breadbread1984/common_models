#!/usr/bin/python3

from absl import flags, app
import requests
import numpy as np
import cv2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = "localhost", help = 'host')
  flags.DEFINE_integer('port', default = 8081, help = 'port')
  flags.DEFINE_string('input', default = 'pics/000035.jpy', help = 'path to picture')

def main(unused_argv):
  img = cv2.imread(FLAGS.input)
  img = img[:,:,::-1]
  img = np.ascontiguousarray(img)
  img = (img / 255).astype(np.float32)
  img = np.transpose(img, (2,0,1))
  inputs = np.expand_dims(img, axis = 0)
  data = {"inputs": {'%inputs': inputs.tolist()}}
  response = requests.post(
    f"http://{FLAGS.host}:{FLAGS.port}",
    headers = {"Content-Type": "application/json"},
    json = data
  )
  assert response.status_code == 200:
  res = response.json()
  import pdb; pdb.set_trace()

if __name__ == "__main__":
  add_options()
  app.run(main)

