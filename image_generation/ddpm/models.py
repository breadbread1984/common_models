#!/usr/bin/python3

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDIMScheduler

class Diffusion(nn.Module):
  def __init__(self, image_size: int = 32, in_channels: int = 3, out_channels: int = 3, timesteps: int = 1000):
    super().__init__()
    self.model = UNet2DModel(
      sample_size = image_size,
      in_channels = in_channels,
      out_channels = out_channels,
      layers_per_block = 2,
      block_out_channels = (128, 128, 256, 256, 512, 512),
      down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
      ),
      up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
      ),
    )
    self.noise_scheduler = DDIMScheduler(num_train_timesteps = timesteps)
    self.image_size = image_size
  def forward(self, noisy_image, timesteps):
    noise_pred = self.model(noisy_image, timesteps).sample # epsilon
    return noise_pred
  def sample(self, batch = 1, num_inference_steps = 50):
    self.noise_scheduler.set_timesteps(num_inference_steps)
    noise = torch.randn((batch,3,self.image_size,self.image_size)).to(next(self.parameters()).device)
    with torch.no_grad():
      for t in tqdm(self.noise_scheduler.timesteps):
        model_output = self.model(noise, t).sample # epsilon(x_t, t)
        noise = self.noise_scheduler.step(model_output, t, noise).prev_sample # x_{t-1}
    image = ((noise / 2 + 0.5).clamp(0, 1) * 255.).to(torch.uint8)
    #image = torch.permute(image, (0,2,3,1)).cpu().numpy()
    return image
