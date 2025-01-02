#!/usr/bin/python3

from absl import flags, app
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import gymnasium_robotics

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('env_num', default = 1, help = 'number of parallel environment')
  flags.DEFINE_string('logdir', default = 'logs', help = 'path to log directory')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epoch')
  flags.DEFINE_integer('episodes', default = 10000, help = 'episodes per epoch')
  flags.DEFINE_float('gamma', default = 0.95, help = 'gamma value')
  flags.DEFINE_float('lam', default = 0.95, help = 'lambda')

def main(unused_argv):
  gym.register_envs(gymnasium_robotics)
  env = gym.make("FetchPickAndPlaceDense-v3", render_mode = "rgb_array")
  for ep in tqdm(range(FLAGS.epochs)):
    for e in tqdm(range(FLAGS.episodes), leave = False):
      states, rewards, dones = list(), list(), list()
      obs, info = env.reset()
      states.append(obs)
      while True:
        action = env.action_space.sample()
        obs, reward, done, truc, info = env.step(action)
        rewards.append(reward)
        dones.append(done)
        if done:
          assert len(rewards) == len(dones)
          rewards = np.array(rewards)
          v_values = discount_cumsum(rewards, gamma = FLAGS.gamma)
          advantages = gae(rewards, v_values, dones, FLAGS.gamma, FLAGS.lam)
          break
        states.append(obs)
      import pdb; pdb.set_trace() 
        
def discount_cumsum(rewards, gamma = 1.):
  discount_cumsum = np.zeros_like(rewards)
  discount_cusum[-1] = rewards[-1]
  for t in reversed(range(rewards.shape[0] - 1)):
    discount_cumsum[t] = rewards[t] + gamma * discount_cumsum[t + 1]
  return discount_cumsum

def gae(rewards, values, dones, gamma, lam):
  T = len(rewards)
  advantages = np.zeros(T)
  advantage = 0
  for t in reversed(range(T)):
    delta = rewards[t] + (gamma * values[t + 1] if not dones[t] else 0) - values[t]
    advantages[t] = delta + (gamma * lam * advantages[t + 1] if not dones[t] else 0)
  return advantages

if __name__ == "__main__":
  add_options()
  app.run(main)

