#!/usr/bin/python3

from typing import Any, Union
from collections.abc import Sequence
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class HFChatPromptTemplate(ChatPromptTemplate):
  def format(self, **kwargs) -> str:
    tokenizer = kwargs.get('tokenizer')
    messages = self.format_messages(**kwargs)
    string_messages = []
    for m in messages:
      if isinstance(m, HumanMessage):
        role = 'user'
      elif isinstance(m, AIMessage):
        role = 'assistant'
      elif isinstance(m, SystemMessage):
        role = 'system'
      else:
        raise Exception(f'Got unsupported message type: {m}')
    messages.append({'role': role, 'content': m.content})
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

if __name__ == "__main__":
  messages = [
    ('system', "you are a helpful AI bot. Your name is {name}"),
    ('human', '{user_input}')
  ]
  prompt = ChatPromptTemplate.from_messages(messages)
  txt = prompt.format(**{'user_input': 'what is your name', 'name': 'robot test'})
  print(str(txt))
  prompt = HFChatPromptTemplate.from_messages(messages)
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  txt = prompt.format(**{'user_input': 'what is your name', 'name': 'robot test'}, tokenizer = tokenizer)
  print(txt)
