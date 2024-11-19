#!/usr/bin/python3

from absl import flags, app
import os
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = 'cropped', help = 'path to preprocessed dataset')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 8, help = 'epochs')
  flags.DEFINE_integer('workers', default = 8, help = 'number of worker')
  flags.DEFINE_float('lr', default = 0.001, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

class CelebA(Dataset):
  def __init__(self, root_path):
    self.paths = list()
    for f in os.listdir(root_path):
      stem, ext = os.path.splitext(f)
      if ext != '.pt': continue
      self.paths.append(join(root_path,f))
  def __len__(self):
    return len(self.paths)
  def __getitem__(self, index):
    sample = torch.load(self.list[index])
    return sample['image'], sample['label']

def main(unused_argv):
  trainset = CelebA(root_path = os.path.join(FLAGS.dataset, 'train'))
  valset = CelebA(root_path = os.path.join(FLAGS.dataset, 'valid'))
  resnet = InceptionResnetV1(
    classify = True,
    pretrained = 'vggface2',
    num_classes = unique_ids.shape[0]).to(FLAGS.device)
  optimizer = optim.Adam(resnet.parameters(), lr = FLAGS.lr)
  scheduler = MultiStepLR(optimizer, [5, 10])
  train_loader = DataLoader(
    trainset,
    num_workers = FLAGS.workers,
    batch_size = FLAGS.batch_size,
  )
  val_loader = DataLoader(
    valset,
    num_workers = FLAGS.workers,
    batch_size = FLAGS.batch_size,
  )
  loss_fn = torch.nn.CrossEntropyLoss()
  metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
  }
  writer = SummaryWriter()
  writer.iteration, writer.interval = 0, 10

  print('\n\nInitial')
  print('-' * 10)
  resnet.eval()
  training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=FLAGS.device,
    writer=writer
  )

  for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=FLAGS.device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=FLAGS.device,
        writer=writer
    )

  writer.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

