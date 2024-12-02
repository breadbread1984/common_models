#!/usr/bin/python3

from functools import partial
from datasets import load_dataset

def tokenize_function(examples, tokenizer):
  return tokenizer(examples['text'], padding = 'max_length', truncation = True, max_length = 512)

def load_imdb(tokenizer):
  train = load_dataset('imdb', split = 'train')
  valid = load_dataset('imdb', split = 'test')
  train = train.map(partial(tokenize_function, tokenizer = tokenizer), batched = True)
  valid = valid.map(partial(tokenize_function, tokenizer = tokenizer), batched = True)
  train = train.rename_column("label", "labels")
  valid = valid.rename_column("label", "labels")
  train.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
  valid.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
  return train, valid

if __name__ == "__main__":
  from huggingface_hub import login
  from transformers import AutoTokenizer
  login('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')
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased') 
  train, valid = load_imdb(tokenizer)
  print(train)
