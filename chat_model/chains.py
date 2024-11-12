#!/usr/bin/python3

from models import TGI
from transformers import AutoTokenizer
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from hf_chat_template import HFChatPromptTemplate

def chat_chain(host):
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  template = HFChatPromptTemplate([
    SystemMessage(content = '你是一个海盗，用海盗应该有的强调回复用户的任何问题。'),
    MessagesPlaceholder('chat_history'),
    HumanMessage('{input}')
  ], tokenizer = tokenizer)
  chain = template | TGI(host)
  return chain
