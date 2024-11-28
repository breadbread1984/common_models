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
  # get user features
  train = Dataset(join(FLAGS.dataste, 'processed', 'train', '*.parquet'), part_size = "500MB")
  user_features = unique_rows_by_features(train, Tags.USER, Tags.USER_ID).compute().reset_index(drop = True)
  user_features.to_parquet(join('feast_repo', 'data', 'user_features.parquet'))
  # load trained two tower model
  model = tf.keras.models.load_model(join(FLAGS.ckpt, 'query_tower'))
  feature = ["user_id", "user_shops", "user_profile", "user_group", "user_gender", "user_age", "user_consumption_2",
             "user_is_occupied", "user_geography", "user_intentions", "user_brands", "user_categories"] >> \
            TransformWorkflow(get_workflow().get_subworkflow("user")) >> \
            PredictTensorflow(model)
  workflow = nvt.Workflow(['user_id'] + feature)
  user_embeddings = workflow.fit_transform(Dataset(user_features)).to_ddf().compute()
  user_embeddings.to_parquet(join('feast_repo', 'data', 'user_embeddings.parquet'))

if __name__ == "__main__":
  add_options()
  app.run(main)
