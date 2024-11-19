#!/usr/bin/python3

from os import mkdir, makedirs
from os.path import exists, join
from absl import flags, app
from tqdm import tqdm
import torch
from torch import nn, autograd, device
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import Diffusion
from create_datasets import load_datasets
from dataclasses import dataclass
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, notebook_launcher

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('processes', default = 1, help = 'number of process')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to the checkpoint')
  flags.DEFINE_integer('batch_size', default = 16, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 50, help = 'number of epochs')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')
  flags.DEFINE_integer('workers', default = 4, help = 'worker number')

def main(unused_argv):
  @dataclass
  class TrainingConfig:
    image_size = 128
    train_batch_size = FLAGS.batch_size
    eval_batch_size = 16
    num_epochs = FLAGS.epochs
    gradient_accumulation_steps = 1
    learning_rate = FLAGS.lr
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = FLAGS.ckpt
    push_to_hub = False
    overwrite_output_dir = True
    seed = 0
  config = TrainingConfig()
  autograd.set_detect_anomaly(True)
  model = Diffusion()
  model.to(device(FLAGS.device))
  trainset = load_datasets(config)
  train_dataloader = DataLoader(trainset, batch_size = config.train_batch_size, shuffle = True, num_workers = FLAGS.workers)
  optimizer = AdamW(model.parameters(), lr = config.learning_rate)
  lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer,
    num_warmup_steps = config.lr_warmup_steps,
    num_training_steps = len(train_dataloader) * config.num_epochs
  )
  args = (config, model, optimizer, train_dataloader, lr_scheduler)
  notebook_launcher(train_loop, args, num_processes = FLAGS.processes)

def evaluate(config, epoch, pipeline):
  images = pipeline(
    batch_size = config.eval_batch_size,
    generator = torch.Generator(device = 'cpu').manual_seed(config.seed)
  ).images
  image_grid = make_image_grid(images, rows = 4, cols = 4)
  test_dir = join(config.output_dir, "samples")
  makedirs(test_dir, exist_ok = True)
  image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
  accelerator = Accelerator(
    mixed_precision = config.mixed_precision,
    gradient_accumulation_steps = config.gradient_accumulation_steps,
    log_with = "tensorboard",
    project_dir = join(config.output_dir, 'summaries')
  )
  if accelerator.is_main_process:
    if config.output_dir is not None:
      makedirs(config.output_dir, exist_ok = True)
    accelerator.init_trackers("train_example")
  model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
  global_step = 0
  for epoch in range(FLAGS.epochs):
    progress_bar = tqdm(total = len(train_dataloader), disable = not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
      clean_images = batch['images']
      noise = torch.randn(clean_images.shape, device = clean_images.device)
      bs = clean_images.shape[0]
      timesteps = torch.randint(
        0, accelerator.unwrap_model(model).noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
        dtype=torch.int64
      )
      noisy_images = accelerator.unwrap_model(model).noise_scheduler.add_noise(clean_images, noise, timesteps)
      with accelerator.accumulate(model):
        noise_pred = model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
          accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    progress_bar.update(1)
    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
    progress_bar.set_postfix(**logs)
    accelerator.log(logs, step = global_step)
    global_step += 1

    # evaluation
    if accelerator.is_main_process:
      pipeline = DDPMPipeline(unet = accelerator.unwrap_model(model).model, scheduler = accelerator.unwrap_model(model).noise_scheduler)
      if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
        evaluate(config, epoch, pipeline)

if __name__ == "__main__":
  add_options()
  app.run(main)

