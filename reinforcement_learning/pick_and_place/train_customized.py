#!/usr/bin/python3

from absl import flags, app
from tqdm import tqdm
import numpy as np
import cv2
import gymnasium as gym
import gymnasium_robotics

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('env_num', default = 1, help = 'number of parallel environment')
  flags.DEFINE_string('logdir', default = 'logs', help = 'path to log directory')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epoch')
  flags.DEFINE_integer('episodes', default = 10000, help = 'episodes per epoch')
  flags.DEFINE_integer('max_ep_steps', default = 300, help = 'max episode steps')
  flags.DEFINE_float('gamma', default = 0.95, help = 'gamma value')
  flags.DEFINE_float('lam', default = 0.95, help = 'lambda')
  flags.DEFINE_boolean('visualize', default = False, help = 'whether to visualize')

def main(unused_argv):
  gym.register_envs(gymnasium_robotics)
  env = gym.make("FetchPickAndPlaceDense-v3", render_mode = "rgb_array")
  for epoch in tqdm(range(FLAGS.epochs)):
    for episode in tqdm(range(FLAGS.episodes), leave = False):
      states, actions, rewards, dones = list(), list(), list(), list()
      obs, info = env.reset()
      states.append(obs['observation']) # s_t
      for step in range(FLAGS.max_ep_steps):
        action = env.action_space.sample() # a_t
        obs, reward, done, truc, info = env.step(action) # r_t
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if FLAGS.visualize:
          image = env.render()[:,:,::-1]
          cv2.imshow("", image)
          cv2.waitKey(1)
          print(reward)
        if done:
          print(len(states), len(reward), len(dones))
          assert len(states) == len(rewards) == len(dones)
          states = np.stack(states) # states.shape = (len, 25)
          actions = np.array(actions) # actions.shape = (len)
          rewards = np.array(rewards) # rewards.shape = (len)
          dones = np.array(dones) # rewards.shape = (len)
          v_values = discount_cumsum(rewards, gamma = FLAGS.gamma)
          advantages = gae(rewards, v_values, dones, FLAGS.gamma, FLAGS.lam)
          break
        states.append(obs['observation'])
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

