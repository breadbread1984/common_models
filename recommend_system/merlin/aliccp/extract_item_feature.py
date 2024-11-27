#!/usr/bin/python3

from absl import flags, app
from merlin.io.dataset import Dataset
import merlin.models.tf as mm
import nvtabular as nvt
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'tt_ckpt', help = 'path to ckpt')

def main(unused_argv):
  # get item features
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  item_features = unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop = True)
  item_features.to_parquet(join(FLAGS.dataset, 'data', 'item_features.parquet'))
  # load trained two tower model
  model = mm.TwoTowerModel(
    train.schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER]).without(['click','conversion'])
    query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),
    samplers=[mm.InBatchSampler()],
    embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
  )
  model.load_weights(join(FLAGS.ckpt, 'twotower_weights.h5'))
  # create feature extraction workflow
  item_id = ["item_id"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsItemID()
  item_features = ["item_category", "item_shop", "item_brand"] >> nvt.ops.Categorify(dtype = "int32") >> nvt.ops.TagAsItemFeatures()
  subgraph_item = Subgraph("item", item_id + item_features) >> nvt.ops.Dropna()
  transform_workflow = nvt.Workflow(subgraph_item)
  feature = ['item_id', 'item_brand', 'item_category', 'item_shop'] >> TransformWorkflow(transform_workflow) >> PredictTensorflow(model.first.item_block())
  workflow = nvt.Workflow(['item_id'] + feature)
  item_embeddings = workflow.fit_transform(Dataset(item_features)).to_ddf().compute()
  item_embeddings.to_parquet(join(FLAGS.dataset, 'data', 'item_embeddings.parquet'))

if __name__ == "__main__":
  add_options()
  app.run(main)
