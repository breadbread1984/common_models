#!/usr/bin/python3

from typing import Any, Union
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

def get_prompt():
  template = ChatPromptTemplate([
    ('system', '你是一个海盗，用海盗应该有的腔调回复用户的任何问题。'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
  ])
  return template
