#!/usr/bin/python3

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.env_util import make_vec_env

def load_fetchpickplace_env(para_num = 1, logdir = "log"):
  gym.register_envs(gymnasium_robotics)
  #env = gym.make("FetchPickAndPlace-v3", render_mode = "rgb_array")
  env = make_vec_env("FetchPickAndPlaceDense-v3", n_envs = para_num, monitor_dir = logdir)
  return env

if __name__ == "__main__":
  import cv2
  env = load_fetchpickplace_env(1)
  observation, info = env.reset(seed = 42)
  for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    image = env.render()[:,:,::-1]
    cv2.imshow("", image)
    cv2.waitKey(20)
  # NOTE: https://robotics.farama.org/envs/fetch/pick_and_place/
  # observation['observation'].shape = (25,)
  # observation['achieved_goal'].shape = (3,)
  # observation['desired_goal'].shape = (3,)
  env.close()
