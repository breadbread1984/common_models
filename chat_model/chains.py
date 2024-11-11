#!/usr/bin/python3

from models import TGI
from prompts import chat_template

def chat_chain(host):
  chain = chat_template() | TGI(host)
  return chain
