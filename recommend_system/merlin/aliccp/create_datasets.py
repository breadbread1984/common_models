#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import join, exists
from absl import flags, app
from pathlib import Path
from merlin.datasets.ecommerce import get_aliccp, transform_aliccp
#from merlin.models.utils.dataset import unique_rows_by_features
from merlin.core.dispatch import get_lib
from merlin.schema.tags import Tags
import nvtabular as nvt
from merlin.dag.ops.subgraph import Subgraph

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('aliccp', default = 'aliccp', help = 'aliccp root directory')

def main(unused_argv):
  if not exists(join(FLAGS.aliccp, 'transformed')):
    get_aliccp(path = FLAGS.aliccp, convert_train = True, convert_test = True)
  transform_datasets(FLAGS.aliccp)

def get_workflow():
  item_id = ["item_id"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsItemID()
  item_features = ["item_category", "item_shop", "item_brand"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsItemFeatures()
  user_id = ["user_id"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsUserID()
  user_features = ["user_shops", "user_profile", "user_group", "user_gender", "user_age", "user_consumption_2",
                   "user_is_occupied", "user_geography", "user_intentions", "user_brands", "user_categories"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsUserFeatures()
  subgraph_item = Subgraph("item", item_id + item_features)
  subgraph_user = Subgraph("user", user_id + user_features)
  targets = ["click", "conversion"] >> nvt.ops.AddMetadata(tags = [Tags.BINARY_CLASSIFICATION, Tags.TARGET])
  outputs = subgraph_user + subgraph_item + targets
  outputs = outputs >> nvt.ops.Dropna()
  outputs.graph.render('workflow.dot')
  workflow = nvt.Workflow(outputs)
  return workflow

def transform_datasets(root_path = 'dataset'):
  #item_features = unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop = True)
  #item_features.to_parquet(join(root_path, 'data', 'item_features.parquet'))
  # define attributes subsets
  workflow = get_workflow()
  transform_aliccp((nvt.Dataset(join(root_path, 'transformed', 'train'), engine = 'parquet'),
                    nvt.Dataset(join(root_path, 'transformed', 'valid'), engine = 'parquet')),
                   Path(join(root_path, 'processed')),
                   nvt_workflow = workflow,
                   workflow_name = "workflow")
  workflow.save('dlrm_torch.workflow')

if __name__ == "__main__":
  add_options()
  app.run(main)

