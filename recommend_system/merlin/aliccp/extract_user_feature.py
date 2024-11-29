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
  flags.DEFINE_integer('batch', default = 1024 * 8, help = 'batch size')

def main(unused_argv):
  # get user features
  train = Dataset(join(FLAGS.dataste, 'processed', 'train', '*.parquet'), part_size = "500MB")
  user_features = unique_rows_by_features(train, Tags.USER, Tags.USER_ID).compute().reset_index(drop = True)
  user_features.to_parquet(join('feast_repo', 'data', 'user_features.parquet'))
  # load trained two tower model
  model = mm.TwoTowerModel(
    train.schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER]).without(['click','conversion']),
    query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),
    samplers=[mm.InBatchSampler()],
    embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
  )
  model.compile(
    optimizer = "adam",
    run_eagerly = False,
    loss = "categorical_crossentropy",
    metrics = [mm.RecallAt(10), mm.NDCGAt(10)]
  )
  model.fit(train, batch_size = FLAGS.batch, epochs = 1, steps_per_epoch = 1)
  model.load_weights(join(FLAGS.ckpt, 'tt_ckpt'))
  # create feature extraction workflow
  feature = ["user_id", "user_shops", "user_profile", "user_group", "user_gender", "user_age", "user_consumption_2",
             "user_is_occupied", "user_geography", "user_intentions", "user_brands", "user_categories"] >> \
            TransformWorkflow(get_workflow().get_subworkflow("user")) >> \
            PredictTensorflow(model.retrieval_block.query_block())
  workflow = nvt.Workflow(['user_id'] + feature)
  user_embeddings = workflow.fit_transform(Dataset(user_features)).to_ddf().compute()
  user_embeddings.to_parquet(join('feast_repo', 'data', 'user_embeddings.parquet'))

if __name__ == "__main__":
  add_options()
  app.run(main)
