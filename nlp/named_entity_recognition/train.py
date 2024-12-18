#!/usr/bin/python3

from absl import flags, app
from functools import partial
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

def compute_metrics(eval_pred, id_to_label):
  predictions, labels = eval_pred # predictions.shape = (batch, seq_len, class_num)
  predictions = np.argmax(predictions, axis = 2)
  # filter ctrl token and padding token
  true_predictions = [
    [id_to_label[p] for (p,l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
  ]
  true_labels = [
    [id_to_label[l] for (p,l) in zip(prediction, label) if l != -100]
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
  ner_labels = train.features['ner_tags'].feature.names
  label_to_id = {label: idx for idx, label in enumerate(ner_labels)}
  id_to_label = {idx: label for label, idx in label_to_id.items()}
  model = BertForTokenClassification.from_pretrained('google-bert/bert-base-uncased' if not FLAGS.eval_only else FLAGS.load_ckpt, num_labels = len(ner_labels), id2label = id_to_label, label2id = label_to_id)
  training_args = TrainingArguments(
    output_dir = FLAGS.save_ckpt,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = FLAGS.lr,
    per_device_train_batch_size = FLAGS.batch,
    per_device_eval_batch_size = FLAGS.batch,
    num_train_epochs = FLAGS.epochs,
    warmup_steps = 500,
    weight_decay = 0.01,
    logging_dir = "./logs",
    logging_steps = 10)
  trainer = Trainer(
    model = model,
    args = training_args, 
    train_dataset = train,
    eval_dataset = valid,
    compute_metrics = partial(compute_metrics, id_to_label = id_to_label))
  if not FLAGS.eval_only:
    trainer.train(resume_from_checkpoint = FLAGS.load_ckpt)
    trainer.save_model('best_model')
  else:
    eval_res = trainer.evaluate()
    print(eval_res)

if __name__ == "__main__":
  add_options()
  app.run(main)

