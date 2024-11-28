#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
import tensorflow as tf
from merlin.io.dataset import Dataset
import merlin.models.tf as mm
import nvtabular as nvt
from merlin.schema.tags import Tags
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
from create_datasets import get_workflow

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'tt_ckpt', help = 'path to ckpt')

def main(unused_argv):
  # get item features
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  item_features = unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop = True)
  item_features.to_parquet(join('feast_repo', 'data', 'item_features.parquet'))
  # load trained two tower model
  model = tf.keras.models.load_model(join(FLAGS.ckpt, 'tt_ckpt'))
  # create feature extraction workflow
  feature = ['item_id', 'item_brand', 'item_category', 'item_shop'] >> \
          TransformWorkflow(get_workflow().get_subworkflow("item")) >> \
          PredictTensorflow(model.first.item_block())
  workflow = nvt.Workflow(['item_id'] + feature)
  item_embeddings = workflow.fit_transform(Dataset(item_features)).to_ddf().compute()
  item_embeddings.to_parquet(join('feast_repo', 'data', 'item_embeddings.parquet'))

if __name__ == "__main__":
  add_options()
  app.run(main)
