#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
import tensorflow as tf
from merlin.io.dataset import Dataset
import merlin.models.tf as mm
from merlin.schema.tags import Tags

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'tt_ckpt', help = 'path to ckpt')
  flags.DEFINE_integer('batch', default = 1024 * 8, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 20, help = 'epochs')

def main(unused_argv):
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
  valid = Dataset(join(FLAGS.dataset, 'processed', 'valid', '*.parquet'), part_size = "500MB")
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
  if exists(FLAGS.ckpt):
    model.load_weights(join(FLAGS.ckpt, 'tt_ckpt'))
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=join(FLAGS.ckpt, 'tt_ckpt'),
    save_weights_only=True,  # 仅保存权重。如果想保存整个模型，设置为 False。
    monitor="val_loss",  # 根据验证集 loss 保存最优模型
    save_best_only=True,  # 只保存最优的模型
    mode="min"
  )
  model.fit(train, validation_data = valid, batch_size = FLAGS.batch, epochs = FLAGS.epochs, callbacks = [checkpoint_callback])
  '''
  query_tower = model.retrieval_block.query_block()
  query_tower.save(join(FLAGS.ckpt, 'query_tower'))
  item_tower = model.retrieval_block.item_block()
  item_tower.save(join(FLAGS.ckpt, 'item_tower'))
  '''

if __name__ == "__main__":
  add_options()
  app.run(main)
