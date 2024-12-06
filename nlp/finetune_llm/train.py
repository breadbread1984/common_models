#!/usr/bin/python3

from absl import flags, app
from torch import device
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import deepspeed
from create_datasets import load_hotpotqa

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('save_ckpt', default = 'ckpt', help = 'path to save checkpoint')
  flags.DEFINE_string('load_ckpt', default = None, help = 'path to load checkpoint')
  flags.DEFINE_float('lr', default = 5e-5, help = 'learning rate')
  flags.DEFINE_integer('epochs', default = 3, help = 'epochs')
  flags.DEFINE_integer('max_seq_length', default = 32768, help = 'max sequence length')
  flags.DEFINE_integer('batch', default = 8, help = 'batch size')
  flags.DEFINE_boolean('eval_only', default = False, help = 'whether to do evaluation only')
  flags.DEFINE_integer('local_rank', default = None, help = 'local_rank')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_integer('dp', default = 1, help = 'data parallel number')
  flags.DEFINE_integer('tp', default = 1, help = 'tensor parallel number')
  flags.DEFINE_integer('pp', default = 1, help = 'pipeline parallel number')

def main(unused_argv):
  ds_configs = {
    "train_micro_batch_size_per_gpu": FLAGS.batch,
    "fp16": {
      "enabled": True,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
      }
    },
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
      },
      "allgather_partitions": True,
      "allgather_bucket_size": 2e8,
      "reduce_scatter": True,
      "reduce_bucket_size": 2e8,
      "overlap_comm": True,
      "load_from_fp32_weights": True,
      "elastic_checkpoint": True
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 2000,
    "wall_clock_breakdown": False,
    "tensor_parallel": {
      "tp_size": FLAGS.tp
    },
    "pipeline_parallel": {
      "pp_size": FLAGS.pp
    },
    "data_parallel": {
      "dp_size": FLAGS.dp
    }
  }
  train, valid = load_hotpotqa()
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code = True)
  model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct' if not FLAGS.eval_only else FLAGS.load_ckpt, trust_remote_code = True)
  model.to(device(FLAGS.device, FLAGS.local_rank))
  ora_peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 16, lora_alpha = 32, lora_dropout = 0.05)
  training_args = SFTConfig(
    output_dir = FLAGS.save_ckpt,
    per_device_train_batch_size = FLAGS.batch,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    weight_decay = 0.01,
    num_train_epochs = FLAGS.epochs,
    logging_dir = "./logs",
    logging_steps = 100,
    gradient_accumulation_steps = 4,
    fp16 = True,
    learning_rate = FLAGS.lr,
    warmup_steps = 500,
    deepspeed = ds_configs,
  )
  trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = train,
    eval_dataset = valid,
    tokenizer = tokenizer,
    max_seq_length = FLAGS.max_seq_length,
  )
  print(f'data parallelism: {deepspeed.utils.get_data_parallel_world_size()}')
  print(f'tensor parallelism: {deepspeed.utils.get_tensor_parallel_world_size()}')
  print(f'pipeline parallelism: {deepspeed.utils.get_pipeline_parallel_world_size()}')
  if not FLAGS.eval_only:
    trainer.train(resume_from_checkpoint = FLAGS.load_ckpt)
    if FLAGS.local_rank == 0:
      trainer.save_model('best_model')
  else:
    eval_res = trainer.evaluate()
    print(eval_res)

if __name__ == "__main__":
  add_options()
  app.run(main)
