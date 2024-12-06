#!/usr/bin/python3

from absl import flags, app
import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.algo import SAC
from torchrl.envs import GymEnvWrapper
from torchrl.envs.wrappers import VecNormalize
from torchrl.agents import ActorCriticAgent
from torchrl.agents.sac import SACAgent
from torchrl.common import set_seed
from torchrl.common.misc import set_device
from torchrl.envs import gym
from create_datasets import load_fetchpickplace_env

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('iters', default = 10000, help = 'training iterations')
  flags.DEFINE_integer('batch', default = 256, help = 'batch size')

def main(unused_argv):
  # 设定随机种子
  set_seed(0)

  # 设定设备
  device = set_device('cuda')

  # 创建环境
  env = GymEnvWrapper(load_fetchpickplace_env())

  # 定义观测和动作空间
  obs_spec = env.observation_space
  act_spec = env.action_space

  # 设定神经网络参数
  actor_lr = 0.001
  critic_lr = 0.001
  actor_hidden_size = 256
  critic_hidden_size = 256

  # 定义SAC代理
  sac_agent = SACAgent(
    env,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    actor_hidden_size=actor_hidden_size,
    critic_hidden_size=critic_hidden_size,
    gamma=0.99,
    tau=0.005,
    batch_size=FLAGS.batch,
    buffer_size=100000,
    update_interval=1,
    num_updates=1,
    device=device
  )

  # 设定回放缓存
  replay_buffer = sac_agent.replay_buffer

  # 训练代理
  num_iterations = FLAGS.iters
  for i in range(num_iterations):
    # 收集数据
    time_step, _ = env.reset()
    trajectory = []
    while not time_step.is_last():
        action = sac_agent.actor(time_step.observation)
        next_time_step = env.step(action)
        trajectory.append(next_time_step)
        time_step = next_time_step
    trajectory = trajectory.concat()

    # 添加数据到回放缓存
    replay_buffer.add(trajectory)

    # 采样数据
    batch = replay_buffer.sample(sac_agent.batch_size)

    # 更新代理
    sac_agent.update(batch)

    # 评估代理
    eval_time_step, _ = env.reset()
    eval_trajectory = []
    while not eval_time_step.is_last():
        action = sac_agent.actor(eval_time_step.observation)
        next_time_step = env.step(action)
        eval_trajectory.append(next_time_step)
        eval_time_step = next_time_step
    eval_trajectory = eval_trajectory.concat()

    # 打印评估奖励
    print(f"Iteration {i+1}, Eval Reward: {eval_trajectory.discount() * eval_trajectory.reward()}")

if __name__ == "__main__":
  add_options()
  app.run(main)

