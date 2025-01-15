#!/usr/bin/python3

from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from models import Llama3_2
from configs import *

class State(TypedDict):
  rank: int
  chat_history: list
  question: str
  generation: str
  documents: List[Document]

def get_graph(k = 5):
  graph_builder = StateGraph(State)
  llm = Llama3_2()
  # create rephrase node
  rephrase_prompt = ChatPromptTemplate.from_messages([
    ('system', """You a question re-writer that converts an input question to a better version that is optimized \n
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
    ('user', "Here is the initial question: \n\n {question} \n Formulate an improved question.")
  ])
  question_rephraser = rephrase_prompt | llm
  def rephrase(state: State):
    rank = state['rank']
    chat_history = state['chat_history']
    question = state['question']
    question = question_rephraser.invoke({'question': question})
    return {'rank': rank, 'chat_history': chat_history, 'question': question}
  graph_builder.add_node('rephrase', rephrase)
  # create retriever node
  embedding = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-base")
  vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db, index_name = "typical_rag")
  retriever = vectordb.as_retriever(search_kwargs = {"k": k})
  def retrieval(state: State):
    rank = state['rank']
    chat_history = state['chat_history']
    question = state['question']
    documents = retriever.invoke(question)
    return {'rank': rank, 'chat_history': chat_history, 'question': question, 'documents': documents}
  graph_builder.add_node("retrieval", retrieval)
  # create filter document node
  def filterdoc(state: State):
    rank = state['rank']
    chat_history = state['chat_history']
    question = state['question']
    documents = state['documents']
    documents = [doc for doc in documents if doc.metadata['classification'] <= rank]
    return {'rank': rank, 'chat_history': chat_history, 'question': question, 'documents': documents}
  graph_builder.add_node("filter", filterdoc)
  # create generation node
  prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer any use questions in the language in which it was entered or required based solely on the context below:\n\n<context>\n{context}\n</context>'),
    MessagesPlaceholder('chat_history'),
    ('user', '{input}')
  ])
  rag_chain = prompt | llm | StrOutputParser()
  def generation(state: State):
    chat_history = state['chat_history']
    documents = state['documents']
    question = state['question']
    generation = rag_chain.invoke({'context': documents, 'chat_history': chat_history, 'input': question})
    return {"chat_history": chat_history, "documents": documents, "question": question, "generation": generation}
  graph_builder.add_node("chatbot", generation)
  # create check hallucination node
  hallucination_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
    ('user', "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
  ])
  hallucination_grader = hallucination_prompt | llm
  answer_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""),
    ('user', "User question: \n\n {question} \n\n LLM generation: {generation}")
  ])
  answer_grader = answer_prompt | llm
  def check_output(state: State):
    documents = state['documents']
    question = state['question']
    generation = state['generation']
    print(generation)
    supported = hallucination_grader.invoke({'generation': generation, 'documents': documents})
    if supported.content.lower() == 'yes':
      resolved = answer_grader.invoke({'question': question, 'generation': generation})
      return 'useful' if resolved.content.lower() == 'yes' else 'not useful'
    else:
      return 'not supported'
  # add edges
  graph_builder.add_edge(START, "retrieval")
  graph_builder.add_edge("retrieval", "filter")
  graph_builder.add_edge("filter", "chatbot")
  graph_builder.add_conditional_edges("chatbot", check_output, {
    'useful': END, # answer the question -> end
    'not useful': 'rephrase', # not answer the question -> rephrase
    'not supported': 'chatbot', # meet hallucination -> regenerate
  })
  graph_builder.add_edge('rephrase', 'retrieval')
  graph = graph_builder.compile()
  with open('graph.mmd', 'w') as f:
    f.write(graph.get_graph().draw_mermaid())
  return graph
