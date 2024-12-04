#!/usr/bin/python3

from absl import flags, app
from wget import download
import json
from chains import rephrase_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('tgi_host', default = 'http://localhost:8080/generate', help = 'tgi host')
  flags.DEFINE_string('output', default = 'dataset.txt', help = 'path to output path')

def main(unused_argv):
  download('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json')
  chain = rephrase_chain(FLAGS.tgi_host)
  with open('train-v2.0.json', 'r') as f:
    dataset = json.loads(f.read())
  for sample in dataset['data']:
    for paragraph in sample['paragraphs']:
      for qa in paragraph['qas']:
          question = qa['question']
          answer = qa['answers']
          sentence = chain.invoke({'question': question, 'answer': answer})
          print(question, answer, sentence)
          exit()

if __name__ == "__main__":
  add_options()
  app.run(main)

