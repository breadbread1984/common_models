#!/usr/bin/python3

from absl import flags, app
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('batch', default = 32, help = 'batch size')
  flags.DEFINE_integer('workers', default = 8, help = 'number of workers')
  flags.DEFINE_boolean('download', default = False, help = 'whether to download celeba')

def main(unused_argv):
  trainset = CelebA(root = 'celeba', split = 'train', download = FLAGS.download)
  evalset = CelebA(root = 'celeba', split = 'valid', download = FLAGS.download)
  trainset_loader = DataLoader(trainset, batch_size = FLAGS.batch, shuffle = True, num_workers = FLAGS.workers)
  evalset_loader = DataLoader(evalset, batch_size = FLAGS.batch, shuffle = True, num_workers = FLAGS.workers)
  for batch in trainset_loader:
    print(batch)

if __name__ == "__main__":
  add_options()
  app.run(main)

