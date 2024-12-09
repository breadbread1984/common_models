#!/usr/bin/python3

from absl import flags, app
from stable_baselines3 import SAC
from create_datasets import load_fetchpickplace_env

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to save checkpoint')
  flags.DEFINE_integer('train_steps', default = 1000000, help = 'total training steps')

def main(unused_argv):
  env = load_fetchpickplace_env()
  model = SAC("MultiInputPolicy", env, verbose = 1)
  model.learn(total_timesteps = FLAGS.train_steps)
  model.save(FLAGS.ckpt)

if __name__ == "__main__":
  add_options()
  app.run(main)

