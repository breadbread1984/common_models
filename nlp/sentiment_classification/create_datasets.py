#!/usr/bin/python3

from datasets import load_dataset

def tokenize_function(examples):
  return tokenizer(examples['text'], padding = 'max_length', truncation = True, max_length = 512)

def load_imdb():
  train = load_dataset('imdb', split = 'train')
  valid = load_dataset('imdb', split = 'test')
  train = train.map(tokenize_function, batched = True)
  valid = valid.map(tokenize_function, batched = True)
  train = train.rename_column("label", "labels")
  valid = valid.rename_column("label", "labels")
  train.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
  valid.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
  return train, valid

if __name__ == "__main__":
  train, valid = load_imdb()
  print(train)
