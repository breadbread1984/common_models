#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
import tensorflow as tf
from merlin.io.dataset import Dataset
import merlin.models.tf as mm
from merlin.schema.tags import Tags
from merlin.systems.dag import Ensemble
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'tt_ckpt', help = 'path to ckpt')
  flags.DEFINE_integer('batch', default = 1024 * 8, help = 'batch size')
  flags.DEFINE_float('lr', default = 5e-3, help = 'learning rate')
  flags.DEFINE_integer('epochs', default = 20, help = 'epochs')
  flags.DEFINE_string('pipeline', default = 'tt_pipeline_tf', help = '')

def main(unused_argv):
  # 1) training two towar model
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  valid = Dataset(join(FLAGS.dataset, 'processed', 'valid', '*.parquet'), part_size = "500MB")
  model = mm.TwoTowerModel(
    train.schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER]).without(['click','conversion']),
    query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),
    samplers=[mm.InBatchSampler()],
    embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
  )
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
  model.compile(
    optimizer = optimizer,
    run_eagerly = False,
    loss = "categorical_crossentropy",
    metrics = [mm.RecallAt(10), mm.NDCGAt(10)]
  )
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=join(FLAGS.ckpt, 'tt_ckpt'), save_weights_only = True, save_freq = 'epoch'),
    tf.keras.callbacks.TensorBoard(log_dir = join(FLAGS.ckpt, 'logs'))
  ]
  model.fit(train, validation_data = valid, batch_size = FLAGS.batch, epochs = FLAGS.epochs, callbacks = callbacks)
  metrics = model.evaluate(valid, batch_size = FLAGS.batch, return_dict = True)
  print(metrics)
  # 2) generate pipeline
  # NOTE: only user feature is calculated online when deployment
  pipeline = ["user_id", "user_shops", "user_profile", "user_group", "user_gender", "user_age", "user_consumption_2",
              "user_is_occupied", "user_geography", "user_intentions", "user_brands", "user_categories"] >> \
             TransformWorkflow(workflow.get_subworkflow('user')) >> \
             PredictTensorflow(model.retrieval_block.query_block())
  ensemble = Ensemble(pipeline, workflow.get_subworkflow('user').input_schema)
  ensemble.export(FLAGS.pipeline)
  # 3) generate item feature and user feature
  workflow = nvt.Workflow.load('dlrm_torch.workflow')
  item_feature = ['item_id', 'item_brand', 'item_category', 'item_shop'] >> \
          TransformWorkflow(workflow.get_subworkflow("item")) >> \
          PredictTensorflow(model.retrieval_block.item_block())
  item_workflow = nvt.Workflow(['item_id'] + item_feature)
  item_embeddings = item_workflow.fit_transform(Dataset(item_features)).to_ddf().compute()
  item_embeddings.to_parquet(join('feast_repo', 'data', 'item_embeddings.parquet'))
  user_feature = ["user_id", "user_shops", "user_profile", "user_group", "user_gender", "user_age", "user_consumption_2",
                  "user_is_occupied", "user_geography", "user_intentions", "user_brands", "user_categories"] >> \
                  TransformWorkflow(get_workflow().get_subworkflow("user")) >> \
                  PredictTensorflow(model.retrieval_block.query_block())
  user_workflow = nvt.Workflow(['user_id'] + user_feature)
  user_embeddings = user_workflow.fit_transform(Dataset(user_features)).to_ddf().compute()
  user_embeddings.to_parquet(join('feast_repo', 'data', 'user_embeddings.parquet'))

  '''
  query_tower = model.retrieval_block.query_block()
  query_tower.save(join(FLAGS.ckpt, 'query_tower'))
  item_tower = model.retrieval_block.item_block()
  item_tower.save(join(FLAGS.ckpt, 'item_tower'))
  '''

if __name__ == "__main__":
  add_options()
  app.run(main)

