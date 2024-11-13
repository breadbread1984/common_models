#!/usr/bin/python3

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

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to the checkpoint')
  flags.DEFINE_integer('batch_size', default = 64, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 600, help = 'number of epochs')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  dist.init_process_group(backend = 'nccl')
  torch.cuda.set_device(dist.get_rank())

