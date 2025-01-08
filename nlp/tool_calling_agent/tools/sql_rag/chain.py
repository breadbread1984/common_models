#!/usr/bin/python3

from langchain_experimental.sql import SQLDatabaseChain
from prompts import sqlite_prompt

def sql_rag(llm, db):
  chain = SQLDatabaseChain.from_llm(llm = llm, db = db, prompt = sqlite_prompt())
  return chain
