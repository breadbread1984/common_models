#!/usr/bin/python3

from os import makedirs
from absl import flags, app
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from create_datasets import load_fetchpickplace_env

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to save checkpoint')
  flags.DEFINE_string('logdir', default = 'logs', help = 'path to log directory')
  flags.DEFINE_integer('train_steps', default = 500000, help = 'total training steps')

def main(unused_argv):
  env = load_fetchpickplace_env()
  makedirs(FLAGS.logdir, exist_ok = True)
  env = Monitor(env, FLAGS.logdir)
  vec_env = make_vec_env(env_id, n_envs=4, monitor_dir=log_dir)
  model = PPO(
    "MultiInputPolicy",  # 使用多层感知机（MLP）策略
    vec_env,      # 向量化环境
    learning_rate = 3e-4,      # 学习率
    n_steps = 2048,            # 每次更新时使用的环境步数
    batch_size = 64,           # 每次梯度下降的批大小
    n_epochs = 10,             # 每次更新的训练回合数
    gamma = 0.99,              # 折扣因子
    gae_lambda = 0.95,         # 广义优势估计的 lambda
    clip_range = 0.2,          # PPO 的裁剪范围
    verbose = 1,               # 输出训练信息
    tensorboard_log = FLAGS.logdir  # TensorBoard 日志目录
  )
  checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=FLAGS.logdir, name_prefix="ppo_fetch")
  eval_callback = EvalCallback(
    vec_env,
    best_model_save_path = "./best_model/",
    log_path = FLAGS.logdir,
    eval_freq = 10000,
    deterministic = True,
    render = False
  )
  model.learn(total_timesteps = FLAGS.train_steps, callback = [checkpoint_callback, eval_callback])
  model.save(FLAGS.ckpt)

if __name__ == "__main__":
  add_options()
  app.run(main)

