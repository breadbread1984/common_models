#!/usr/bin/python3

from models import TGI, Test
from transformers import AutoTokenizer
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from hf_chat_template import HFChatPromptTemplate

def chat_chain(host):
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  template = HFChatPromptTemplate([
    SystemMessage(content = '你是一个海盗，用海盗应该有的腔调回复用户的任何问题。'),
    MessagesPlaceholder('chat_history'),
    HumanMessage('{input}')
  ], tokenizer = tokenizer)
  chain = template | TGI(host)
  return chain

if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  template = HFChatPromptTemplate([
    SystemMessage(content = '你是一个海盗，用海盗应该有的腔调回复用户的任何问题。'),
    MessagesPlaceholder('chat_history'),
    HumanMessage('{input}')
  ], tokenizer = tokenizer)
  chain = template | Test()
  print(chain.invoke({'input':'你最喜欢的音乐是什么','chat_history':[]}))
