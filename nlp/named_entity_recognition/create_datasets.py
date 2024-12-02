#!/usr/bin/python3

from functools import partial
from datasets import load_dataset

def tokenize_function(examples, tokenizer):
  datasets = tokenizer(examples['tokens'], truncation = True, is_split_into_words = True, padding = 'max_length', max_length = 512)
  labels = list()
  for i, label in enumerate(examples['tags']):
    word_ids = tokenized_inputs.word_ids(batch_index = i)
    previous_word_idx = None
    label_ids = list()
    for word_idx in word_ids:
      if word_idx is None:
        label_ids.append(-100)
      elif word_idx != previous_word_idx:
        label_ids.append(label[word_idx])
      else:
        label_ids.append(label[previous_word_idx])
      previous_word_idx = word_idx
    labels.append(label_ids)
  datasets['labels'] = labels
  return datasets

def load_conll2003(tokenizer):
  train = load_dataset('conll2003', split = 'train', trust_remote_code = True)
  valid = load_dataset('conll2003', split = 'validation', trust_remote_code = True)
  train = train.map(partial(tokenize_function, tokenizer = tokenizer), batched = True)
  valid = valid.map(partial(tokenize_function, tokenizer = tokenizer), batched = True)

if __name__ == "__main__":
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
  load_conll2003(tokenizer)
