#!/usr/bin/python3

from absl import flags, app
import torch
from torch import device, load
from models import Generator
import cv2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to the checkpoint')
  flags.DEFINE_integer('img_size', default = 32, help = 'image size')
  flags.DEFINE_integer('dim', default = 100, help = 'hidden dimension')
  flags.DEFINE_integer('batch', default = 1, help = 'how many images are sampled')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  generator = Generator(img_size = FLAGS.img_size, latent_dim = FLAGS.dim).to(device(FLAGS.device))
  z = torch.normal(mean = 0, std = 1, size = (FLAGS.batch, FLAGS.dim)).to(next(generator.parameters()).device)
  fake_imgs = generator(z)
  imgs = ((fake_imgs / 2 + 0.5).clamp(0,1) * 255.).to(torch.uint8)
  imgs = torch.permute(imgs, (0,2,3,1)).detch().cpu().numpy()
  for idx, img in enumerate(imgs):
    cv2.imwrite(f'{idx}.png', img)

if __name__ == "__main__":
  add_options()
  app.run(main)
