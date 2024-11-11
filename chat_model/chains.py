#!/usr/bin/python3

from models import TGI
from prompts import chat_template

def chat_chain():
  chain = chat_template() | TGI()
  return chain
