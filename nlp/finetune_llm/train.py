#!/usr/bin/python3

from absl import flags, app
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import DeepSpeedEngine, DeepSpeedConfig
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

def main(unused_argv):
  configs = {
    "fp16": {
      "enabled": true,
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
        "pin_memory": true
      },
      "allgather_partitions": true,
      "allgather_bucket_size": 2e8,
      "reduce_scatter": true,
      "reduce_bucket_size": 2e8,
      "overlap_comm": true,
      "load_from_fp32_weights": true,
      "elastic_checkpoint": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
  }
  train, valid = load_hotpotqa()
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code = True)
  model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code = True)
  ds_config = DeepSpeedConfig(configs)
  ora_peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 16, lora_alpha = 32, lora_dropout = 0.05)
  trainer = SFTTrainer(
    output_dir = FLAGS.save_ckpt,
    evaluation_strategy = "epoch",
    save_strateg = "epoch",
    model = model,
    train_dataset = train,
    eval_dataset = valid,
    max_seq_length = FLAGS.max_seq_length,
    tokenizer = tokenizer,
    per_device_train_batch_size = FLAGS.batch,
    per_device_eval_batch_size = FLAGS.batch,
    gradient_accumulation_steps = 4,
    deepspeed = ds_config,
    num_train_epochs = FLAGS.epochs,
    logging_dir = "./logs",
    logging_steps = 100,
  )
  if not FLAGS.eval_only:
    trainer.train(resume_from_checkpoint = FLAGS.load_ckpt)
    trainer.save_model('best_model')
  else:
    eval_res = trainer.evaluate()
    print(eval_res)

if __name__ == "__main__":
  add_options()
  app.run(main)

