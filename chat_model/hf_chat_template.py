#!/usr/bin/python3

from typing import Any, Union
from collections.abc import Sequence
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# NOTE: langchain-core >= 0.3.15

class HFChatPromptTemplate(ChatPromptTemplate):
  tokenizer: Any = None
  def format(self, **kwargs) -> str:
    messages = self.format_messages(**kwargs)
    hf_messages = []
    for m in messages:
      if isinstance(m, HumanMessage):
        role = 'user'
      elif isinstance(m, AIMessage):
        role = 'assistant'
      elif isinstance(m, SystemMessage):
        role = 'system'
      else:
        raise Exception(f'Got unsupported message type: {m}')
      hf_messages.append({'role': role, 'content': m.content})
    return self.tokenizer.apply_chat_template(hf_messages, tokenize = False, add_generation_prompt = True)

if __name__ == "__main__":
  messages = [
    ('system', "you are a helpful AI bot. Your name is {name}"),
    MessagesPlaceholder('chat_history'),
    ('human', '{user_input}')
  ]
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  prompt = HFChatPromptTemplate(messages, tokenizer = tokenizer)
  txt = prompt.format(**{'user_input': 'what is your name', 'name': 'robot test', 'chat_history': [HumanMessage(content = 'your name is awesome!'), AIMessage(content = 'Thanks!')]})
  print(txt)
