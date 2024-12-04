#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
from wget import download
import json
from chains import rephrase_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('tgi_host', default = 'http://localhost:8080/generate', help = 'tgi host')
  flags.DEFINE_string('output', default = 'dataset.txt', help = 'path to output path')

def main(unused_argv):
  if not exists('train-v2.0.json'):
    download('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json')
  chain = rephrase_chain(FLAGS.tgi_host)
  with open('train-v2.0.json', 'r') as f:
    dataset = json.loads(f.read())
  with open(FLAGS.output, 'w') as f:
    for sample in dataset['data']:
      for paragraph in sample['paragraphs']:
        for qa in paragraph['qas']:
          question = qa['question']
          answer = qa['answers'][0]['text']
          sentence = chain.invoke({'question': question, 'answer': answer})
          f.write(sentence + '\n')

if __name__ == "__main__":
  add_options()
  app.run(main)

