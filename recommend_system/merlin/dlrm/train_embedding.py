#!/bin/bash/python3

from absl import flags, app
from merlin.core.dispatch import get_lib
from merlin.schema.tags import Tags
import nvtabular as nvt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset')

def load_datasets(root_path):
  train = get_lib().read_parquet(join(root_path, 'ml-1m', 'train.parquet'))
  valid = get_lib().read_parquet(join(root_path, 'ml-1m', 'valid.parquet'))
  # table head: userId 	movieId 	rating 	timestamp
  train_ds = nvt.Dataset(train)
  valid_ds = nvt.Dataset(valid)
  return train_ds, valid_ds

def main(unused_argv):
  output = ['userId', 'movieId'] >> nvt.ops.Categorify()
  output += ['rating'] >> nvt.ops.AddMetadata(tags = [Tags.REGRESSION, Tags.TARGET])
  output.graph.render(filename = "graph.dot")
  # NOTE: generate png with "dot -Tps graph.dot -o graph.png"
  workflow = nvt.Workflow(output)

if __name__ == "__main__":
  add_options()
  app.run(main)
