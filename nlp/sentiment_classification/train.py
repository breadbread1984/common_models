#!/usr/bin/python3

from absl import flags, app
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from create_datasets import load_imdb

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('save_ckpt', default = 'ckpt', help = 'path to save checkpoint')
  flags.DEFINE_string('load_ckpt', default = None, help = 'path to resume checkpoint')
  flags.DEFINE_integer('batch', default = 128, help = 'batch size')
  flags.DEFINE_float('lr', default = 2e-4, help = 'learning rate')
  flags.DEFINE_integer('epochs', default = 3, help = 'number of epochs')
  flags.DEFINE_boolean('eval_only', default = False, help = 'only do evaluation')

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = torch.argmax(torch.tensor(logits), dim = -1)
  precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average = "binary")
  acc = accuracy_score(labels, predictions)
  return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
  model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased' if not FLAGS.eval_only else FLAGS.load_ckpt, num_labels = 2)
  training_args = TrainingArguments(
    output_dir = FLAGS.save_ckpt,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = FLAGS.lr,
    per_device_train_batch_size = FLAGS.batch,
    per_device_eval_batch_size = FLAGS.batch,
    num_train_epochs = FLAGS.epochs,
    weight_decay = 0.01,
    logging_dir = "./logs",
    logging_steps = 10)
  train, valid = load_imdb(tokenizer)
  trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train,
    eval_dataset = valid,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics)
  if not FLAGS.eval_only:
    trainer.train(resume_from_checkpoint = FLAGS.load_ckpt)
    trainer.save_model('best_model')
  else:
    eval_res = trainer.evaluate()
    print(eval_res)

if __name__ == "__main__":
  add_options()
  app.run(main)

