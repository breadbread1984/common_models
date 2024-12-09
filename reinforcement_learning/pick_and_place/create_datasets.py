#!/usr/bin/python3

import gymnasium as gym
import gymnasium_robotics

def load_fetchpickplace_env():
  gym.register_envs(gymnasium_robotics)
  env = gym.make("FetchPickAndPlace-v3", render_mode = "rgb_array")
  return env

if __name__ == "__main__":
  import cv2
  env = load_fetchpickplace_env()
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
