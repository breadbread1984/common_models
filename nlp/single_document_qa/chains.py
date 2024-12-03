#!/usr/bin/python3

from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from prompts import load_map_rerank_prompt, load_stuff_prompt, load_map_reduce_prompt, load_refine_prompt

def map_rerank_chain(tokenizer, llm, **kwargs):
  prompt = load_map_rerank_prompt(tokenizer)
  chain = load_qa_with_sources_chain(llm, chain_type = 'map_rerank', prompt = prompt, **kwargs)
  return chain

def stuff_chain(tokenizer, llm, **kwargs):
  prompt, document_prompt = load_stuff_prompt(tokenizer)
  chain = load_qa_with_sources_chain(llm, chain_type = 'stuff', prompt = prompt, document_prompt = document_prompt, **kwargs)
  return chain

def map_reduce_chain(tokenizer, llm, **kwargs):
  question_prompt, combine_prompt, document_prompt = load_map_reduce_prompt(tokenizer)
  chain = load_qa_with_sources_chain(llm, chain_type = 'map_reduce', question_prompt = question_prompt, combine_prompt = combine_prompt, document_prompt = document_prompt, **kwargs)
  return chain

def refine_chain(tokenizer, llm, **kwargs):
  question_prompt, refine_prompt, document_prompt = load_refine_prompt(tokenizer)
  chain = load_qa_with_sources_chain(llm, chain_type = 'refine', question_prompt = question_prompt, refine_prompt = refine_prompt, document_prompt = document_prompt)
  return chain

if __name__ == "__main__":
  from transformers import AutoTokenizer
  from models import TGI
  llm = TGI('http://localhost:8080')
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  chain = map_rerank_chain(tokenizer, llm)
  chain = stuff_chain(tokenizer, llm)
  chain = map_reduce_chain(tokenizer, llm)
  chain = refine_chain(tokenizer, llm)
