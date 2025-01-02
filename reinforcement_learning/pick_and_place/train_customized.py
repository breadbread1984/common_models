#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
from tqdm import tqdm
import numpy as np
import cv2
import gymnasium as gym
import gymnasium_robotics
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('env_num', default = 1, help = 'number of parallel environment')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'checkpoint')
  flags.DEFINE_string('logdir', default = 'logs', help = 'path to log directory')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epoch')
  flags.DEFINE_integer('episodes', default = 10000, help = 'episodes per epoch')
  flags.DEFINE_integer('max_ep_steps', default = 300, help = 'max episode steps')
  flags.DEFINE_float('gamma', default = 0.95, help = 'gamma value')
  flags.DEFINE_float('lam', default = 0.95, help = 'lambda')
  flags.DEFINE_boolean('visualize', default = False, help = 'whether to visualize')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')

class PolicyNet(nn.Module):
  def __init__(self, state_dim = 25, action_dim = 4, hidden_dim = 16):
    super(PolicyNet, self).__init__()
    self.mean = nn.Sequential(
      nn.Linear(state_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, action_dim),
      nn.Tanh()
    )
    self.log_std = nn.Parameter(torch.zeros(1, action_dim))
  def forward(self, x):
    mean = self.mean(x) # mean.shape = (batch, action_dim)
    std = torch.tile(torch.exp(self.log_std), (mean.shape[0], 1)) # std.shape = (batch, action_dim)
    return mean, std

class ValueNet(nn.Module):
  def __init__(self, state_dim = 25, hidden_dim = 16):
    super(ValueNet, self).__init__()
    self.valuenet = nn.Sequential(
      nn.Linear(state_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1)
    )
  def forward(self, x):
    values = self.valuenet(x)
    return values

class PPO(nn.Module):
  def __init__(self, state_dim = 25, action_dim = 4, hidden_dim = 16):
    super(PPO, self).__init__()
    self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim)
    self.value_net = ValueNet(state_dim, hidden_dim)
    self.dist = torch.distributions.Normal
  def act(self, x):
    mean, std = self.policy_net(x)
    dist = self.dist(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim = -1)
    return action, log_prob
  def advantages(self, states, rewards, values, dones, gamma = 0.95, lam = 0.95):
    assert states.shape[0] == rewards.shape[0] + 1 == values.shape[0] + 1 == dones.shape[0] + 1
    T = len(rewards)
    advantages = torch.zeros(T)
    advantage = 0
    for t in reversed(range(T)):
      delta = rewards[t] + (0 if dones[t] else \
                            gamma * values[t + 1] if t != T - 1 else \
                            gamma * self.value_net(states[-1:])[0,0]) - values[t]
      advantages[t] = delta + (0 if dones[t] else \
                               gamma * lam * advantages[t + 1] if t != T - 1 else \
                               0)
    assert advantages.shape[0] == rewards.shape[0]
    return advantages
  def get_values(self, states, rewards, dones, gamma):
    assert states.shape[0] == rewards.shape[0] + 1 == dones.shape[0] + 1
    # calculate the V(s_t) of the last state s_t before truncation
    discount_cumsum = torch.zeros_like(rewards).to(next(self.parameters()).device) # discount_cumsum.shape = (len)
    discount_cumsum[-1] = rewards[-1] + (0 if dones[-1] else gamma * self.value_net(states[-1:])[0,0])
    # calculate the leading V(s_t)
    for t in reversed(range(rewards.shape[0] - 1)):
      discount_cumsum[t] = rewards[t] + gamma * discount_cumsum[t + 1]
    assert discount_cumsum.shape[0] == rewards.shape[0]
    return discount_cumsum
  def pred_values(self, states):
    values = self.value_net(states[:-1]) # values.shape = (len, 1)
    values = torch.squeeze(values, dim = -1) # values.shape = (len,)
    return values

def main(unused_argv):
  gym.register_envs(gymnasium_robotics)
  env = gym.make("FetchPickAndPlaceDense-v3", render_mode = "rgb_array")
  ppo = PPO()
  criterion = nn.MSELoss()
  optimizer = Adam(ppo.parameters(), lr = FLAGS.lr)
  tb_writer = SummaryWriter(log_dir = FLAGS.logdir)
  global_steps = 0
  if exists(FLAGS.ckpt):
    ckpt = torch.load(FLAGS.ckpt)
    global_steps = ckpt['global_steps']
    ppo.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
  for epoch in tqdm(range(FLAGS.epochs)):
    for episode in tqdm(range(FLAGS.episodes), leave = False):
      states, logprobs, rewards, dones = list(), list(), list(), list()
      obs, info = env.reset()
      states.append(obs['observation']) # s_t
      for step in range(FLAGS.max_ep_steps):
        obs = torch.from_numpy(np.expand_dims(obs['observation'], axis = 0).astype(np.float32)).to(next(ppo.parameters()).device) # obs.shape = (1, 25)
        action, logprob = ppo.act(obs) # action.shape = (1,4), logprob.shape = (1,1)
        obs, reward, done, truc, info = env.step(action.detach().squeeze(dim = 0).cpu().numpy()) # r_t
        logprobs.append(logprob)
        rewards.append(reward)
        dones.append(done)
        if FLAGS.visualize:
          image = env.render()[:,:,::-1]
          cv2.imshow("", image)
          cv2.waitKey(1)
          print(reward)
        if done or step == FLAGS.max_ep_steps - 1:
          # episodes are truncated to at most 300 steps
          assert len(states) == len(rewards) == len(dones)
          states.append(obs['observation']) # save s_t+1
          states = torch.from_numpy(np.stack(states).astype(np.float32)).to(next(ppo.parameters()).device) # states.shape = (len + 1, 25)
          logprobs = torch.cat(logprobs, dim = 0).to(next(ppo.parameters()).device) # logprobs.shape = (len)
          rewards = torch.from_numpy(np.array(rewards).astype(np.float32)).to(next(ppo.parameters()).device) # rewards.shape = (len)
          dones = torch.from_numpy(np.array(dones).astype(np.float32)).to(next(ppo.parameters()).device) # rewards.shape = (len)
          true_values = ppo.get_values(states, rewards, dones, gamma = FLAGS.gamma) # true_values.shape = (len)
          pred_values = ppo.pred_values(states) # pred_values.shape = (len)
          advantages = ppo.advantages(states, rewards, true_values, dones, FLAGS.gamma, FLAGS.lam) # advantages.shape = (len)
          break
        states.append(obs['observation'])
      # update policy and value networks
      optimizer.zero_grad()
      loss = -torch.mean(logprobs * advantages) + 0.5 * criterion(pred_values, true_values)
      loss.backward()
      optimizer.step()
      tb_writer.add_scalar('loss', loss, global_steps)
      global_steps += 1
    scheduler.step()
    ckpt = {
      'global_steps': global_steps,
      'state_dict': ppo.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    torch.save(ckpt, FLAGS.ckpt)
  env.close()
        
if __name__ == "__main__":
  add_options()
  app.run(main)

