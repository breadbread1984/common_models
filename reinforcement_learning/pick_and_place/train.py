#!/usr/bin/python3

from absl import flags, app
from stable_baselines3 import SAC
from create_datasets import load_fetchpickplace_env
import cv2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to save checkpoint')
  flags.DEFINE_integer('train_steps', default = 1000000, help = 'total training steps')
  flags.DEFINE_integer('eval_steps', default = '1000', help = 'total evaluation steps')

def main(unused_argv):
  env = load_fetchpickplace_env()
  model = SAC("MultiInputPolicy", env, verbose = 1)
  model.learn(total_timesteps = FLAGS.train_steps)
  model.save(FLAGS.ckpt)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(FLAGS.eval_steps):
    action, states = model.predict(obs, deterministic = True)
    obs, reward, done, info = vec_env.step(action)
    img = vec_env.render()[:,:,::-1]
    cv2.imshow("pick and place", img)
    cv2.waitKey(20)
  env.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

