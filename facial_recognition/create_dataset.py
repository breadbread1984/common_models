#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import join, exists
from absl import flags, app
from tqdm import tqdm
import torch
from torchvision.datasets import LFWPeople
from facenet_pytorch import MTCNN

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('output_dir', default = 'cropped', help = 'output directory')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  trainset = LFWPeople(root = 'lfw', split = 'train', download = True)
  valset = LFWPeople(root = 'lfw', split = 'test', download = True)
  if exists(FLAGS.output_dir):
    rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  mkdir(join(FLAGS.output_dir, 'train'))
  mkdir(join(FLAGS.output_dir, 'valid'))
  mtcnn = MTCNN(
    image_size = 160, margin = 0, min_face_size = 20,
    thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True,
    device = FLAGS.device)
  unique_ids = torch.unique(torch.Tensor(trainset.targets, dtype = torch.int32))
  for sample_id, (image, label) in enumerate(tqdm(trainset)):
    x_aligned = mtcnn(image)
    if x_aligned is None:
      print('no face detected skiped!')
      continue
    idx = torch.argmax((label == unique_ids).to(torch.int32), dim = 0)
    torch.save({'image': x_aligned, 'label': idx}, join(FLAGS.output_dir, 'train', f"{sample_id}.pt"))
  for sample_id, (image, label) in enumerate(tqdm(valset)):
    x_aligned = mtcnn(image)
    if x_aligned is None:
      print('no face detected skiped!')
      continue
    idx = torch.argmax((label == unique_ids).to(torch.int32), dim = 0)
    torch.save({'image': x_aligned, 'label': idx}, join(FLAGS.output_dir, 'valid', f"{sample_id}.pt"))

if __name__ == "__main__":
  add_options()
  app.run(main)
