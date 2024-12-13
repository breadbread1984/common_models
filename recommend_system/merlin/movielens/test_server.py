#!/usr/bin/python3

from absl import flags, app
from tqdm import tqdm
import numpy as np
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
  total = 0
  correct = 0
  for df in valid.to_iter():
    for i in tqdm(range(len(df))):
      userId = df['userId'][i]
      movieId = df['movieId'][i]
      binary_rating = df['binary_rating'][i]
      feeds = [
        httpclient.InferInput("userId", userId.shape, "INT64"),
        httpclient.InferInput("movieId", movieId.shape, "INT64")
      ]
      feeds[0].set_data_from_numpy(np.array([userId], dtype = np.int64))
      feeds[1].set_data_from_numpy(np.array([movieId], dtype = np.int64))
      outputs = [httpclient.InferRequestedOutput("binary_rating/binary_output")]
      response = client.infer("executor_model", inputs = feeds, outputs = outputs, model_version = "1")
      pred = response.as_numpy("binary_rating/binary_output")
      if pred == binary_rating:
        correct += 1
      total += 1
  print(f"accuracy: {correct/total}")

if __name__ == "__main__":
  add_options()
  app.run(main)

