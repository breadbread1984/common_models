#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import join, exists
from absl import flags, app
from merlin.datasets.ecommerce import get_aliccp, transform_aliccp
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.core.dispatch import get_lib
from merlin.schema.tags import Tags
import nvtabular as nvt
from merlin.dag.ops.subgraph import Subgraph

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('aliccp', default = 'aliccp', help = 'aliccp root directory')

def main(unused_argv):
  get_aliccp(path = FLAGS.dataset, convert_train = True, convert_test = True)
  transform_datasets(FLAGS.dataset)

def transform_datasets(root_path = 'dataset'):
  train_raw = get_lib().read_parquet(join(root_path, 'transformed', 'train.parquet'))
  valid_raw = get_lib().read_parquet(join(root_path, 'transformed', 'valid.parquet'))
  item_features = unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop = True)
  item_features.to_parquet(join(root_path, 'data', 'item_features.parquet'))
  # define attributes subsets
  items = ["item_id", "item_category", "item_shop", "item_brand"] >> nvt.ops.Categorify(dtype = "int32")
  item_id = items["item_id"] >> nvt.ops.TagAsItemID() # equals to >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsItemID()
  item_features = items["item_category", "item_shop", "item_brand"] >> nvt.ops.TagAsItemFeatures()
  user_id = ["user_id"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsUserID()
  user_features = ["user_shops", "user_profile", "user_group", "user_gender", "user_age", "user_consumption_2",
                   "user_is_occupied", "user_geography", "user_intentions", "user_brands", "user_categories"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsUserFeatures()
  subgraph_item = Subgraph("item", item_id + item_features)
  subgraph_user = Subgraph("user", user_id + user_features)
  targets = ["click"] >> nvt.ops.AddMetadata(tags = [Tags.BINARY_CLASSIFICATION, Tags.TARGET])
  outputs = subgraph_user + subgraph_item + targets
  outputs = outputs >> nvt.ops.Dropna()
  workflow = nvt.Workflow(outputs)
  workflow.plot('workflow.dot')
  transform_sliccp((train_raw, valid_raw), join(root_path, 'processed'), nvt_workflow = workflow, workflow_name = "workflow")

if __name__ == "__main__":
  add_options()
  app.run(main)

