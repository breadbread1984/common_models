#!/usr/bin/python3

from langchain.schema.output_parser import StrOutputParser
from prompts import condense_question_prompt

def condense_chain(tokenizer, llm):
  chain = condense_question_prompt(tokenizer) | llm | StrOutputParser()
  return chain
