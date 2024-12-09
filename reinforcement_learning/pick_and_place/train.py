#!/usr/bin/python3

from absl import flags, app
from stable_baselines3 import SAC
from create_datasets import load_fetchpickplace_env

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to save checkpoint')
  flags.DEFINE_integer('train_steps', default = 1000000, help = 'total training steps')
  flags.DEFINE_integer('eval_steps', default = '1000', help = 'total evaluation steps')

def main(unused_argv):
  env = load_fetchpickplace_env()
  model = SAC("MlpPolicy", env, verbose = 1)
  model.learn(total_timesteps = FLAGS.train_steps)
  model.save(FLAGS.ckpt)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(FLAGS.eval_steps):
    action, states = model.predict(obs, deterministic = True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
  env.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

