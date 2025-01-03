#!/usr/bin/python3

import torch
from torch import nn

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

