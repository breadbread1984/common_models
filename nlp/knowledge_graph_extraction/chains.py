#!/usr/bin/python3

from models import TGI
from prompts import extract_triplets_template

def extract_kg_chain(host, node_labels, rel_types, relationship_type):
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  template = extract_triplets_template(tokenizer, node_labels, rel_types, relationship_type)
  llm = TGI(host)
  chain = template | llm
  return chain

