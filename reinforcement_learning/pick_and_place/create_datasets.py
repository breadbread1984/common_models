#!/usr/bin/python3

import gymnasium as gym
import gymnasium_robotics

def load_fetchpickplace_env():
  gym.register_envs(gymnasium_robotics)
  env = gym.make("FetchPickAndPlace-v3", render_mode = "rgb_array")
  return env

if __name__ == "__main__":
  env = load_fetchpickplace_env()
  observation, info = env.reset(seed = 42)
  image = env.render()[:,:,::-1]
  import cv2
  cv2.imshow("", image)
  cv2.waitKey()
  # NOTE: https://robotics.farama.org/envs/fetch/pick_and_place/
  # observation['observation'].shape = (25,)
  # observation['achieved_goal'].shape = (3,)
  # observation['desired_goal'].shape = (3,)
  env.close()
