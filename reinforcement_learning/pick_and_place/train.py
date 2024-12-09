#!/usr/bin/python3

from absl import flags, app
from stable_baseline3 import SAC
from create_datasets import load_fetchpickplace_env

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to save checkpoint')

def main(unused_argv):
  env = load_fetchpickplace_env()
  model = SAC("picke&place policy", env. verbose = 1)
  model.learn(total_timesteps = 10000)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
    action, states = model.predict(obs, deterministic = True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
  env.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

