#!/usr/bin/python3

from absl import flags, app
import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from create_datasets import load_conll2003

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('save_ckpt', default = 'ckpt', help = 'path to save checkpoint')
  flags.DEFINE_string('load_ckpt', default = None, help = 'path to resume checkpoint')
  flags.DEFINE_integer('batch', default = 128, help = 'batch size')
  flags.DEFINE_float('lr', default = 2e-4, help = 'learning rate')
  flags.DEFINE_integer('epochs', default = 3, help = 'number of epochs')
  flags.DEFINE_boolean('eval_only', default = False, help = 'only do evaluation')

def compute_metrics(eval_pred):
  predictions, labels = eval_pred # predictions.shape = (batch, seq_len, class_num)
  predictions = np.argmax(predictions, axis = 2)
  # filter ctrl token and padding token
  true_predictions = [
    [label_list[p] for (p,l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
  ]
  true_lables = [
    [label_list[l] for (p,l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
  ]
  metric = evaluate.load('seqeval')
  results = metric.compute(predictions=true_predictions, references=true_labels)
  return {
    "precision": results["overall_precision"],
    "recall": results["overall_recall"],
    "f1": results["overall_f1"],
    "accuracy": results["overall_accuracy"],
  }

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
  train, valid = load_conll2003(tokenizer)
  model = BertForTokenClassification.from_pretrained('google-bert/bert-base-cased', num_labels = len(train.features['ner_tags'].feature.names))
  training_args = TrainingArguments(
    output_dir = FLAGS.save_ckpt,
    num_train_epochs = FLAGS.epochs,
    per_device_train_batch_size = FLAGS.batch,
    per_device_eval_batch_size = FLAGS.batch,
    warmup_steps = 500,
    weight_decay = 0.01,
    logging_dir = "./logs",
    logging_steps = 10)
  trainer = Trainer(
    model = model,
    args = training_args, 
    training_dataset = train,
    eval_dataset = valid,
    compute_metrics = compute_metrics)
  if not FLAGS.eval_only:
    trainer.train()
    trainer.save_model('best_model')
  else:
    eval_res = trainer.evaluate()
    print(eval_res)

if __name__ == "__main__":
  add_options()
  app.run(main)

