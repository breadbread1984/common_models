#!/usr/bin/python3

from absl import flags, app
import gymnasium as gym
import gymnasium_robotics
import cv2
from modesl import PPO

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt.pt', help = 'path to checkpoint')
  flags.DEFINE_integer('max_ep_steps', default = 300, help = 'max episode steps')

def main(unused_argv):
  gym.register_envs(gymnasium_robotics)
  env = gym.make("FetchPickAndPlaceDense-v3", render_mode = "rgb_array")
  ppo = PPO()
  obs, info = env.reset()
  for step in range(FLAGS.max_ep_steps):
    obs = torch.from_numpy(np.expand_dims(obs['observation'], axis = 0).astype(np.float32)).to(next(ppo.parameters()).device) # obs.shape = (1, 25)
    action, logprob = ppo.act(obs) # action.shape = (1,4), logprob.shape = (1,1)
    obs, reward, done, truc, info = env.step(action.detach().squeeze(dim = 0).cpu().numpy()) # r_t
    image = env.render()[:,:,::-1]
    cv2.imshow("", image)
    cv2.waitKey(20)

if __name__ == "__main__":
  add_options()
  app.run(main)
