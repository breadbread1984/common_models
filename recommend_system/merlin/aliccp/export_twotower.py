#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
import merlin.models.tf as mm
from merlin.io.dataset import Dataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'aliccp', help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'tt_ckpt', help = 'path to ckpt')
  flags.DEFINE_integer('batch', default = 1024 * 8, help = 'batch size')

def main(unused_argv):
  train = Dataset(join(FLAGS.dataset, 'processed', 'train', '*.parquet'), part_size = "500MB")
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

if __name__ == "__main__":
  add_options()
  app.run(main)

