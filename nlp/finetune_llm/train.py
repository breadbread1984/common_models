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
  flags.DEFINE_integer('max_seq_length', default = 4900, help = 'max sequence length')
  flags.DEFINE_integer('batch', default = 2, help = 'batch size')
  flags.DEFINE_boolean('eval_only', default = False, help = 'whether to do evaluation only')
  flags.DEFINE_integer('workers', default = 4, help = 'number of workers')
  flags.DEFINE_integer('local_rank', default = None, help = 'local_rank')
  flags.DEFINE_integer('dp', default = 1, help = 'data parallel number')
  flags.DEFINE_integer('tp', default = 1, help = 'tensor parallel number')
  flags.DEFINE_integer('pp', default = 1, help = 'pipeline parallel number')

def main(unused_argv):
  ds_configs = {
    "train_batch_size": 'auto',
    "fp16": {
      "enabled": True,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "zero_optimization": {
      "stage": 3,
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
      },
      "offload_param": {
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
  lora_peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 16, lora_alpha = 32, lora_dropout = 0.05)
  training_args = SFTConfig(
    output_dir = FLAGS.save_ckpt,
    logging_dir = "./logs",
    max_seq_length = FLAGS.max_seq_length,
    deepspeed = ds_configs,
    fp16 = True,
  )
  training_args.set_training(
    learning_rate = FLAGS.lr,
    batch_size = FLAGS.batch,
    weight_decay = 0.01,
    num_epochs = FLAGS.epochs,
    gradient_accumulation_steps = 4
  )
  training_args.set_evaluate(
    strategy = 'epoch',
  )
  training_args.set_save(
    strategy = "epoch",
  )
  training_args.set_logging(
    steps = 100,
    report_to = ["tensorboard"],
  )
  training_args.set_optimizer(
    name = "adamw_torch",
    learning_rate = FLAGS.lr,
    weight_decay = 0.01
  )
  training_args.set_lr_scheduler(
    name = 'linear',
    warmup_steps = 500
  )
  training_args.set_dataloader(
    train_batch_size = FLAGS.batch,
    eval_batch_size = FLAGS.batch,
    num_workers = FLAGS.workers,
    pin_memory = True,
  )
  trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = train,
    eval_dataset = valid,
    tokenizer = tokenizer,
    peft_config = lora_peft_config
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
