#!/usr/bin/python3

from typing import Any, Union, List, Tuple, Optional
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_experimental.graph_transformers.llm import create_unstructured_prompt

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

def extract_triplets_template(tokenizer,
                              node_types: Optional[List[str]] = None,
                              rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None):
  import langchain_experimental
  assert langchain_experimental.__version__ >= '0.3.3'
  chat_prompt = create_unstructured_prompt(node_types, rel_types, relationship_type = 'tuple')
  chat_prompt = HFChatPromptTemplate(chat_prompt.messages, tokenizer = tokenizer)
  return chat_prompt

if __name__ == "__main__":
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  prompt = extract_triplets_template(tokenizer,
                                     node_types = ['electrolyte', 'conductivity', 'precursor'],
                                     rel_types = [
                                       ('electrolyte', 'has_conductivity', 'conductivity'),
                                       ('electrolyte', 'has_precursor', 'precursor')
                                     ])
  print(prompt)

