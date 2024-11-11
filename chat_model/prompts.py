#!/usr/bin/python3

from transformers import AutoTokenizer
from langchain_core.prompts.prompt import PromptTemplate

def chat_template():
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  messages = [
    {'role': 'system', 'content': '你是一个海盗，用海盗应该有的强调回复用户的任何问题。'},
    {'role': 'user', 'content': '{input}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['input'])
  return template
