#!/usr/bin/python3

from typing import Any, Union, List, Tuple, Optional
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain import hub

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

def rephrase_prompt(tokenizer):
  rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
  chat_prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = rephrase_prompt)
    ],
    tokenizer = tokenizer
  )
  return chat_prompt

def retrieval_qa_prompt(tokenizer):
  retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
  qa_prompt = HFChatPromptTemplate(
    messages = retrieval_qa_chat_prompt.messages,
    tokenizer = tokenizer
  )
  return qa_prompt

if __name__ == "__main__":
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  prompt = rephrase_prompt(tokenizer)
  print(prompt)
  prompt = retrieval_qa_prompt(tokenizer)
  print(prompt)
