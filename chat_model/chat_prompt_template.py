#!/usr/bin/python3

from typing import Any, Union
from collections.abc import Sequence
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation

class HFChatPromptTemplate(ChatPromptTemplate):
  tokenizer: object = None
  def __init__(self,
               messages: Sequence[MessageLikeRepresentation],
               *,
               tokenizer: Any):
    super(HFChatPromptTemplate, self).__init__(messages)
    self.tokenizer = tokenizer
  @classmethod
  def from_messages(cls, messages, tokenizer):
    return cls(messages, tokenizer = tokenizer)
  def format_prompt(self, **kwargs):
    messages = self.format_message(**kwargs)
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
    return self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

if __name__ == "__main__":
  prompt = HFChatPromptTemplate.from_messages([
    ('system', "you are a helpful AI bot. Your name is {name}"),
    ('human', '{user_input}')
  ], tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'))
  txt = prompt.invoke({'user_input': 'what is your name', 'name': 'robot test'})
  print(txt)
