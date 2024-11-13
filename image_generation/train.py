#!/usr/bin/python3

from os import mkdir
from os.path import exists, join
from absl import flags, app
import torch
from torch import nn
from torch import device, save, load, no_grad, autograd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter
from models import Diffusion
from datasets import load_datasets

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to the checkpoint')
  flags.DEFINE_integer('batch_size', default = 64, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 600, help = 'number of epochs')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def run_epoch(epoch, model, optimizer, criterion, tb_writer):
  # random split
  trainset, valset = load_datasets()
  trainset_sampler = distributed.DistributedSampler(trainset)
  valset_sampler = distributed.DistributedSampler(valset)
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = trainset_sampler, pin_memory = False)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = evalset_sampler, pin_memory = False)
  # training
  train_dataloader.sampler.set_epoch(epoch)
  model.train()
  for step, (x, label) in tqdm(enumerate(train_dataloader)):
    optimizer.zero_grad()
    x = x.to(device(FLAGS.device))
    t = torch.randint(0, model.noise_scheduler.num_train_timesteps, (x.size(0),), device = device(FLAGS.device))
    noise = torch.randn_like(x)
    x_noisy = model.noise_scheduler.add_noise(x, noise, t)
    output = model(x_noisy, t)
    loss = criterion(output, noise)
    loss.backward()
    optimizer.step()
    global_steps = epoch * len(train_dataloder) + step
    if global_steps % 100 == 0 and dist.get_rank() == 0:
      print('Step #%d Epoch #%d: loss %f, lr %f' % (global_steps, epoch, loss, scheduler.get_last_lr()[0]))
      tb_writer.add_scalar('loss', loss, global_steps)

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  dist.init_process_group(backend = 'nccl')
  torch.cuda.set_device(dist.get_rank())
  model = Diffusion()
  model.to(device(FLAGS.device))
  model = DDP(model, device_ids = [dist.get_rank()], output_device = dist.get_rank(), find_unused_parameters = True)
  criterion = nn.MSELoss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  if dist.get_rank() == 0:
    if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
    tb_writer = SummaryWriter(log_dir = join(FLAG.ckpt, 'summaries'))
  start_epoch = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs):
    run_epoch(epoch, model, optimizer, criterion, tb_writer)
    if dist.get_rank() == 0:
      ckpt = {
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler
      }
      save(ckpt, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()
