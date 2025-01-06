#!/usr/bin/python3

from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def get_prompt():
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
  template = ChatPromptTemplate([
    ('system', '你是一个海盗，用海盗应该有的腔调回复用户的任何问题。'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
  ], tokenizer = tokenizer)
  return template
