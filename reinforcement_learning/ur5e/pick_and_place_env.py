import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import os

class UR5ePickAndPlaceEnv(MujocoEnv):
    def __init__(self, model_path, frame_skip=5):
        # 初始化 UR5e 模型路径
        assert os.path.exists(model_path), f"Model file {model_path} does not exist!"
        self.goal = np.array([0.6, 0.2, 0.5])  # 设置目标位置

        # 初始化 Mujoco 环境
        super().__init__(model_path=model_path, frame_skip=frame_skip)

    def reset_model(self):
        """
        重置环境状态，包括机械臂的位置和速度。
        """
        # 初始化关节角度和速度
        qpos = self.init_qpos + np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        self.set_state(qpos, qvel)

        # 随机初始化物体位置
        object_x = np.random.uniform(0.4, 0.8)
        object_y = np.random.uniform(-0.2, 0.2)
        self.data.set_joint_qpos("object:joint", [object_x, object_y, 0.05, 1, 0, 0, 0])

        return self._get_obs()

    def _get_obs(self):
        """
        获取观测数据，包括机械臂末端位置、物体位置和目标位置。
        """
        end_effector_pos = self.data.site_xpos[self.model.site_name2id("ee_site")]
        object_pos = self.data.body_xpos[self.model.body_name2id("object")]
        return np.concatenate([end_effector_pos, object_pos, self.goal])

    def compute_reward(self):
        """
        计算基于目标距离的奖励。
        """
        end_effector_pos = self.data.site_xpos[self.model.site_name2id("ee_site")]
        object_pos = self.data.body_xpos[self.model.body_name2id("object")]
        distance_to_object = np.linalg.norm(end_effector_pos - object_pos)
        distance_to_goal = np.linalg.norm(object_pos - self.goal)

        # 奖励 = 抓住物体的奖励 + 将物体移动到目标的奖励
        reward = -distance_to_object - distance_to_goal
        return reward

    def step(self, action):
        """
        执行动作，并返回新的观测数据、奖励、是否结束等信息。
        """
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward()
        done = False  # UR5e 抓取任务通常没有结束条件
        return obs, reward, done, False, {}

# 测试抓取环境
if __name__ == "__main__":
    # 替换为你的 UR5e 模型文件路径
    ur5e_model_path = "universal_robots_ur5e/ur5e.xml"

    # 创建环境
    env = UR5ePickAndPlaceEnv(model_path=ur5e_model_path)

    # 重置环境
    obs = env.reset()
    print("Initial Observation:", obs)

    # 执行随机动作
    for _ in range(100):
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, _, _ = env.step(action)
        print(f"Reward: {reward}")
        env.render()
