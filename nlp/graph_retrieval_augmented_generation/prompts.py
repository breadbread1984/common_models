#!/usr/bin/python3

from typing import Any, Union, List, Tuple, Optional
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_experimental.graph_transformers.llm import create_unstructured_prompt
from langchain_neo4j.chains.graph_qa.prompts import CYPHER_QA_PROMPT, CYPHER_GENERATION_PROMPT
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
import config

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

def qa_prompt(tokenizer):
  prompt = HFChatPromptTemplate(
    messages = [
      MessagesPlaceholder('chat_history'),
      HumanMessagePromptTemplate(prompt = CYPHER_QA_PROMPT)
    ],
    tokenizer = tokenizer
  )
  return prompt

def cypher_prompt(tokenizer):
  prompt = HFChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = CYPHER_GENERATION_PROMPT)
    ],
    tokenizer = tokenizer
  )
  return prompt

def fewshot_cypher_prompt(tokenizer, with_selector = False):
  example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
  )
  if with_selector:
    example_selector = SemanticSimilarityExampleSelector.from_examples(
      config.examples,
      HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
      Neo4jVector,
      k = 5,
      input_keys = ["question"],
    )
    prompt = FewShotPromptTemplate(
      example_selector = example_selector,
      example_prompt = example_prompt,
      prefix = "You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
      suffix="User input: {question}\nCypher query: ",
      input_variables=["question", "schema"],
    )
  else:
    prompt = FewShotPromptTemplate(
      examples = config.examples[:5],
      example_prompt = example_prompt,
      prefix = "You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
      suffix="User input: {question}\nCypher query: ",
      input_variables=["question", "schema"],
    )
  return prompt

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
  prompt = qa_prompt(tokenizer)
  print(prompt)
  prompt = cypher_prompt(tokenizer)
  print(prompt)
  prompt = fewshot_cypher_prompt(tokenizer)
  print(prompt)
  prompt = fewshot_cypher_prompt(tokenizer, True)
  print(prompt)