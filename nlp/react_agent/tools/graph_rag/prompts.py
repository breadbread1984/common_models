#!/usr/bin/python3

from typing import Any, Union, List, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_experimental.graph_transformers.llm import create_unstructured_prompt
from langchain_neo4j.chains.graph_qa.prompts import CYPHER_QA_PROMPT, CYPHER_GENERATION_PROMPT
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from .config import node_types, examples

def extract_triplets_template(node_types: Optional[List[str]] = None,
                              rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None):
  import langchain_experimental
  assert langchain_experimental.__version__ >= '0.3.3'
  chat_prompt = create_unstructured_prompt(node_types, rel_types, relationship_type = 'tuple')
  chat_prompt = ChatPromptTemplate(chat_prompt.messages)
  return chat_prompt

def qa_prompt():
  prompt = ChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = CYPHER_QA_PROMPT)
    ]
  )
  return prompt

def cypher_prompt():
  prompt = ChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = CYPHER_GENERATION_PROMPT)
    ]
  )
  return prompt

def fewshot_cypher_prompt(with_selector = False, neo4j_host = 'bolt://localhost:7687', neo4j_user = 'neo4j', neo4j_password = 'neo4j', neo4j_db = 'neo4j'):
  example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
  )
  if with_selector:
    # only add similar examples selected by retriever
    example_selector = SemanticSimilarityExampleSelector.from_examples(
      examples,
      embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
      vectorstore_cls = Neo4jVector,
      k = 5,
      input_keys = ["question"],
      url = neo4j_host,
      username = neo4j_user,
      password = neo4j_password,
      database = neo4j_db,
      index_name = "cypher_example_rag",
    )
    prompt = FewShotPromptTemplate(
      example_selector = example_selector,
      example_prompt = example_prompt,
      prefix = "You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
      suffix="User input: {question}\nCypher query: ",
      input_variables=["question", "schema"],
    )
  else:
    # add leading examples
    prompt = FewShotPromptTemplate(
      examples = examples[:5],
      example_prompt = example_prompt,
      prefix = "You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
      suffix="User input: {question}\nCypher query: ",
      input_variables=["question", "schema"],
    )
  prompt = ChatPromptTemplate(
    messages = [
      HumanMessagePromptTemplate(prompt = prompt)
    ]
  )
  return prompt

