#!/usr/bin/python3

from typing import Any, Union
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain.chains.question_answering.map_rerank_prompt import PROMPT as MAP_RERANK_PROMPT
from langchain.chains.qa_with_sources import map_reduce_prompt, refine_prompts, stuff_prompt

class HFChatPromptValue(ChatPromptValue):
  tokenizer: Any = None
  def to_string(self) -> str:
    hf_messages = []
    for m in self.messages:
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

class HFChatPromptTemplate(ChatPromptTemplate):
  tokenizer: Any = None
  def format_prompt(self, **kwargs: Any) -> PromptValue:
    messages = self.format_messages(**kwargs)
    return HFChatPromptValue(messages = messages, tokenizer = self.tokenizer)
  async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
    messages = await self.format_messages(**kwargs)
    return HFChatPromptValue(messages = messages, tokenizer = self.tokenizer)

def load_map_rerank_prompt(tokenizer):
  prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = MAP_RERANK_PROMPT)
    ],
    tokenizer = tokenizer
  )
  return prompt

def load_stuff_prompt(tokenizer):
  prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = stuff_prompt.PROMPT)
    ],
    tokenizer = tokenizer
  )
  document_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = stuff_prompt.EXAMPLE_PROMPT)
    ],
    tokenizer = tokenizer
  )
  return prompt, document_prompt

def load_map_reduce_prompt(tokenizer):
  question_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = map_reduce_prompt.QUESTION_PROMPT)
    ],
    tokenizer = tokenizer
  )
  combine_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = map_reduce_prompt.COMBINE_PROMPT)
    ],
    tokenizer = tokenizer
  )
  document_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = map_reduce_prompt.EXAMPLE_PROMPT)
    ],
    tokenizer = tokenizer
  )
  return question_prompt, combine_prompt, document_prompt

def load_refine_prompt(tokenizer):
  question_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = refine_prompts.DEFAULT_TEXT_QA_PROMPT)
    ],
    tokenizer = tokenizer
  )
  refine_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = refine_prompts.DEFAULT_REFINE_PROMPT)
    ],
    tokenizer = tokenizer
  )
  document_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = refine_prompts.EXAMPLE_PROMPT)
    ]
  )
  return question_prompt, refine_prompt, document_prompt

if __name__ == "__main__":
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  prompt = load_map_rerank_prompt(tokenizer)
  print(prompt)
  prompt, document_prompt = load_stuff_prompt(tokenizer)
  print(prompt, document_prompt)
  question_prompt, combine_prompt, document_prompt = load_map_reduce_prompt(tokenizer)
  print(question_prompt, combine_prompt, document_prompt)
  question_prompt, refine_prompt, document_prompt = load_refine_prompt(tokenizer)
  print(question_prompt, refine_prompt, document_prompt)
