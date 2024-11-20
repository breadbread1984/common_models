#!/usr/bin/python3

from shutil import rmtree
from os.path import join, exists
from absl import flags, app
from merlin.datasets.entertainment import get_movielens
from merlin.core.dispatch import get_lib
from merlin.schema.tags import Tags
import nvtabular as nvt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'output directory')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  get_movielens(variant="ml-1m", path = FLAGS.output_dir)

def load_datasets(root_path):
  # 0) load original dataset
  train = get_lib().read_parquet(join(root_path, 'ml-1m', 'train.parquet'))
  valid = get_lib().read_parquet(join(root_path, 'ml-1m', 'valid.parquet'))
  # table head: userId  movieId     rating  timestamp
  train_ds = nvt.Dataset(train)
  valid_ds = nvt.Dataset(valid)
  # 1) create preprocess workflow
  # id may not be in integer format, change id to category
  output = ['userId', 'movieId'] >> nvt.ops.Categorify()
  # tag label for regression task
  output += ['rating'] >> nvt.ops.AddMetadata(tags = [Tags.REGRESSION, Tags.TARGET])
  output.graph.render(filename = "graph.dot")
  # NOTE: generate png with "dot -Tps graph.dot -o graph.png"
  workflow = nvt.Workflow(output)
  # 2) preprocess dataset
  # fit category on trainset and save to directory train
  workflow.fit_transform(train_ds).to_parquet('train')
  # apply category map on valset and save to directory valid
  workflow.transform(valid_ds).to_parquet('valid')
  # 3) reload preprocessed dataset
  train_transformed = nvt.Dataset('train', engine = 'parquet')
  valid_transformed = nvt.Dataset('valid', engine = 'parquet')
  return train_transformed, valid_transformed

if __name__ == "__main__":
  add_options()
  app.run(main)
