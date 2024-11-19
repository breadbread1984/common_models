#!/usr/bin/python3

from os import mkdir
from os.path import exists, join
from absl import flags, app
from tqdm import tqdm
import torch
from torch import nn
from torch import device, save, load, no_grad, autograd
from torch.optim import Adam
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision
from models import Generator, Discriminator
from datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to the checkpoint')
  flags.DEFINE_integer('batch_size', default = 64, help = 'batch size')
  flags.DEFINE_integer('img_size', default = 32, help = 'image size')
  flags.DEFINE_integer('dim', default = 100, help = 'hidden dimension')
  flags.DEFINE_integer('n_critic', default = 5, help = 'number of training steps for dscriminator per iter')
  flags.DEFINE_integer('epochs', default = 200, help = 'number of epochs')
  flags.DEFINE_integer('workers', default = 4, help = 'number of workers')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def compute_gradient_penalty(D, real_samples, fake_samples):
  alpha = torch.rand((real_samples.shape[0], 1, 1, 1))
  interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
  interpolates.requires_grad = True
  d_interpolates = D(interpolates)
  gradients = autograd.grad(outputs = d_interpolates, inputs = interpolates, create_graph = True, retain_graph = True, only_inputs = True)[0]
  gradients = torch.reshape(gradients, (gradients.shape[0], -1))
  gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
  return gradient_penalty

def train_one_epoch(epoch, train_dataloader, generator, discriminator, optimizer_G, optimizer_D, tb_writer):
  for step, (real_imgs, label) in tqdm(enumerate(train_dataloader)):
    # train discriminator
    optimizer_D.zero_grad()
    z = torch.normal(mean = 0, std = 1, size = (real_imgs.shape[0], FLAGS.dim)).to(next(generator.parameters()).device)
    fake_imgs = generator(z)
    real_validity = discriminator(real_imgs)
    fake_validity = discriminator(fake_imgs)
    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
    d_loss.backward()
    optimizer_D.step()
    # train generator
    optimizer_G.zero_grad()
    if step % FLAGS.n_critic == 0:
      fake_imgs = generator(z)
      fake_validity = discriminator(fake_imgs)
      g_loss = -torch.mean(fake_validity)
      g_loss.backward()
      global_steps = epoch * len(train_dataloader) + step
      tb_writer.add_scalar('D loss', d_loss.item(), global_steps)
      tb_writer.add_scalar('G loss', g_loss.item(), global_steps)
  imgs = ((fake / 2 + 0.5).clamp(0,1) * 255.).to(torch.uint8)
  grid = torchvision.utils.make_grid(imgs[:9], nrow = 3)
  tb_writer.add_image('fake', grid, global_steps)

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  generator = Generator(img_size = FLAGS.img_size, latent_dim = FLAGS.dim).to(device(FLAGS.device))
  discriminator = Discriminator(img_size = FLAGS.img_size).to(device(FLAGS.device))
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
    train_one_epoch(epoch, train_dataloader, generator, discriminator, optimizer_G, optimizer_D, tb_writer)
  torch.save(generator.state_dict(), 'genenerator.pt')
  torch.save(discriminator.state_dict(), 'discriminator.pt')

if __name__ == "__main__":
  add_options()
  app.run(main)
