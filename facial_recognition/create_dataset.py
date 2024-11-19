#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import join, exists
from absl import flags, app
from torchvision.datasets import CelebA
from facenet_pytorch import MTCNN

FLAGS = flags.FLAGS

def add_option():
  flags.DEFINE_string('output_dir', default = 'cropped', help = 'output directory')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  trainset = CelebA(root = 'celeba', split = 'train', target_type = 'identity', download = True)
  valset = CelebA(root = 'celeba', split = 'valid', target_type = 'identity', download = True)
  if exists(FLAGS.output_dir):
    rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  mkdir(join(FLAGS.output_dir, 'train'))
  mkdir(join(FLAGS.output_dir, 'valid'))
  mtcnn = MTCNN(
    image_size = 160, margin = 0, min_face_size = 20,
    thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True,
    device = FLAGS.device)
  unique_ids = torch.unique(trainset.identity)
  for sample_id, (image, label) in enumerate(trainset):
    x_aligned = mtcnn(image)
    if x_aligned is None:
      print('no face detected skiped!')
      continue
    idx = torch.argmax((label == unique_ids).to(torch.int32), dim = 0)
    torch.save(join(FLAGS.output_dir, 'train', f"{sample_id}.pt"), {'image': x_aligned, 'label': idx})
  for sample_id, (image, label) in enumerate(valset):
    x_aligned = mtcnn(image)
    if x_aligned is None:
      print('no face detected skiped!')
      continue
    idx = torch.argmax((label == unique_ids).to(torch.int32), dim = 0)
    torch.save(join(FLAGS.output_dir, 'valid', f"{sample_id}.pt"), {'image': x_aligned, 'label': idx})

if __name__ == "__main__":
  add_options()
  app.run(main)
