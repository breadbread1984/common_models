#!/usr/bin/python3

from absl import flags, app
from stable_baselines3 import PPO
from create_datasets import load_fetchpickplace_env
import cv2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to load checkpoint')

def main(unused_argv):
  env = load_fetchpickplace_env()
  model = PPO('MultiInputPolicy', env, verbose = 1)
  model.load(FLAGS.ckpt)

  for i in range(10):
    obs, info = env.reset()
    while True:
      action, states = model.predict(obs, deterministic = True)
      obs, reward, done, trunc, info = env.step(action)
      img = env.render()[:,:,::-1]
      cv2.imshow('pick and place', img)
      cv2.waitKey(20)
      if done or trunc: break
  env.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

