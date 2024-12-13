#!/usr/bin/python3

from shutil import rmtree
from os.path import join, exists
from absl import flags, app
from merlin.datasets.entertainment import get_movielens
from merlin.core.dispatch import get_lib
from merlin.schema.tags import Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.triton.export import export
import nvtabular as nvt
from dask.distributed import Client
from create_cluster import load_cluster

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'output directory')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  get_movielens(variant="ml-1m", path = FLAGS.output_dir)

def load_datasets(root_path, n_part = 2, use_cluster = False, export = False):
  # 0) load original dataset
  '''
  train = get_lib().read_parquet(join(root_path, 'ml-1m', 'train.parquet'))
  valid = get_lib().read_parquet(join(root_path, 'ml-1m', 'valid.parquet'))
  '''
  data = get_lib().read_parquet(join(root_path, 'ml-1m', 'train.parquet')).sample(frac=1)
  train = data.iloc[:600_000]
  valid = data.iloc[600_000:]
  movies = get_lib().read_parquet(join(root_path, 'ml-1m', 'movies_converted.parquet'))
  # train and valid are in DataFrame format
  # table head: userId  movieId     rating  timestamp
  train_ds = nvt.Dataset(train, npartitions = n_part) # use npartitions to reduce memory usage for processing one chunk
  valid_ds = nvt.Dataset(valid)
  train_ds.shuffle_by_keys('userId')
  valid_ds.shuffle_by_keys('userId')
  # 1) create preprocess workflow
  if use_cluster:
    cluster = load_cluster()
    client = Client(cluster)
  # id may not be in integer format, change id to category
  genres = ['movieId'] >> nvt.ops.JoinExternal(movies, on = 'movieId', columns_ext = ['movieId', 'genres'])
  genres = genres >> nvt.ops.Categorify(freq_threshold = 10) # convert and filt
  binary_rating = ['rating'] >> nvt.ops.LambdaOp(lambda col: col > 3) >> nvt.ops.Rename(name = 'binary_rating')
  binary_rating = binary_rating >> nvt.ops.AddTags(tags=[Tags.TARGET, Tags.BINARY_CLASSIFICATION])
  userId = ['userId'] >> nvt.ops.Categorify() >> nvt.ops.AddTags(tags = [Tags.USER_ID, Tags.CATEGORICAL, Tags.USER])
  movieId = ['movieId'] >> nvt.ops.Categorify() >> nvt.ops.AddTags(tags = [Tags.ITEM_ID, Tags.CATEGORICAL, Tags.ITEM])
  workflow = nvt.Workflow(userId + movieId + genres + binary_rating)
  # 2) preprocess dataset
  # fit category on trainset and save to directory train
  workflow.fit_transform(train_ds).to_parquet('train')
  # apply category map on valset and save to directory valid
  workflow.transform(valid_ds).to_parquet('valid')
  if export:
    ensemble = Ensemble(workflow.input_schema, workflow.output_schema)
    ensemble.add_workflow(workflow)
    export(ensemble, 'model_repo', name = "nvt_workflow")
  # 3) reload preprocessed dataset
  train_transformed = nvt.Dataset('train', engine = 'parquet')
  valid_transformed = nvt.Dataset('valid', engine = 'parquet')
  return train_transformed, valid_transformed

if __name__ == "__main__":
  add_options()
  app.run(main)
