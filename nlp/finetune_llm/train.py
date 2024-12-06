#!/usr/bin/python3

from absl import flags, app
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

def main(unused_argv):
  ds_configs = {
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
        "lr": FLAGS.lr,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 5e-5,
        "warmup_num_steps": 500
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
    "train_batch_size": FLAGS.batch,
    "eval_batch_size": FLAGS.batch,
    "gradient_clipping": 1.0,
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
  }
  train, valid = load_hotpotqa()
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code = True)
  model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct' if not FLAGS.eval_only else FLAGS.load_ckpt, trust_remote_code = True)
  ora_peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 16, lora_alpha = 32, lora_dropout = 0.05)
  training_args = SFTConfig(
    output_dir = FLAGS.save_ckpt,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = FLAGS.lr,
    per_device_train_batch_size = FLAGS.batch,
    per_device_eval_batch_size = FLAGS.batch,
    num_train_epochs = FLAGS.epochs,
    logging_dir = "./logs",
    logging_steps = 100,
    gradient_accumulation_steps = 4,
    deepspeed = ds_configs,
    data_text_field = "messages",
  )
  trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = train,
    eval_dataset = valid,
    tokenizer = tokenizer,
    max_seq_length = FLAGS.max_seq_length,
  )
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

