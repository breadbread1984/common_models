#!/usr/bin/python3

from os import mkdir
from os.path import exists, join
from absl import flags, app
import torch
from torch import nn
from torch import device, save, load, no_grad, autograd
from torch.optim import Adam
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Discriminator
from datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to the checkpoint')
  flags.DEFINE_integer('batch_size', default = 64, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 200, help = 'number of epochs')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def train_one_epoch(epoch, train_dataloader, generator, discriminator, optimizer_G, optimizer_D, tb_writer):
  for step, (x, label) in tqdm(enumerate(train_dataloader)):
    

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  generator = Generator().to(device(FLAGS.device))
  discriminator = Discriminator().to(device(FLAGS.device))
  trainset = load_datasets()
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.workers)
  optimizer_G = Adam(generator.parameters(), lr = FLAGS.lr)
  optimizer_D = Adam(discriminator.parameters(), lr = FLAGS.lr)
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.path'))
    generator.load_state_dict(ckpt['gen_state_dict'])
    discriminator.load_state_dict(ckpt['dis_state_dict'])
    optimizer_G.load_state_dict(ckpt['optimizer_G'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs):
    generator.train()
    discriminator.train()
