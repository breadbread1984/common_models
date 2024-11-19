#!/usr/bin/python3

from absl import flags, app
import os
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.datasets import CelebA
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 8, help = 'epochs')
  flags.DEFINE_integer('workers', default = 8, help = 'number of worker')
  flags.DEFINE_float('lr', default = 0.001, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

class CropFace(object):
  def __init__(self, ):
    self.mtcnn = MTCNN(
      image_size = 160, margin = 0, min_face_size = 20,
      thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True,
      device = FLAGS.device)
  def __call__(self, image):
    x_aligned = self.mtcnn(image)
    return x_aligned

class Identity(object):
  def __init__(self, unique_ids):
    self.unique_ids = unique_ids
  def __call__(self, label):
    idx = torch.argmax((label == self.unique_ids).to(torch.int32), dim = 0)
    return idx

def main(unused_argv):
  trainset = CelebA(root = 'celeba', split = 'train', target_type = 'identity', download = True)
  trans = transforms.Compose([
    CropFace(),
  ])
  unique_ids = torch.unique(trainset.identity)
  target_trans = transforms.Compose([
    Identity(unique_ids),
  ])
  trainset = CelebA(root = 'celeba', split = 'train', target_type = 'identity', download = True, transform = trans, target_transform = target_trans)
  valset = CelebA(root = 'celeba', split = 'valid', target_type = 'identity', download = True, transform = trans, target_transform = target_trans)
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

