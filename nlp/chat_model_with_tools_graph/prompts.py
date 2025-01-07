#!/usr/bin/python3

from langchain import hub
from typing import Any, Union
from transformers import AutoTokenizer
from langchain.tools.render import render_text_description
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue

# NOTE: langchain-core >= 0.3.15

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

def agent_template(tokenizer):
  prompt = hub.pull("hwchase17/openai-functions-agent")
  assert len(prompt) == 4
  prompt = HFChatPromptTemplate(
    prompt,
    tokenizer = tokenizer
  )
  return prompt
