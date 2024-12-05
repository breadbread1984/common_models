#!/usr/bin/python3

from gymnasium_robotics.envs.fetch.fetch_env import MujocoFetchEnv

class UR5ePickAndPlaceEnv(MujocoFetchEnv):
  def __init__(self, reward_type = 'sparse'):
    initial_qpos = {
      'robot0:slide0': 0.405,
      'robot0:slide1': 0.48,
      'robot0:slide2': 0.0,
      'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    super(UR5ePickAndPlaceEnv, self).__init__(
      model_path = './universal_robots_ur5e/ur5e.xml', has_object=True, block_gripper=False,
      n_substeps=20, gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
      obj_range=0.15, target_range=0.15, distance_threshold=0.05,
      initial_qpos=initial_qpos, reward_type=reward_type)

if __name__ == "__main__":
  env = UR5ePickAndPlaceEnv()
  obs = env.reset()
  env.render()
