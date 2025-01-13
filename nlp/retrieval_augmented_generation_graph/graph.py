#!/usr/bin/python3

from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from models import Llama3_2
from configs import *

class State(TypedDict):
  question: str
  generation: str
  documents: List[str]

def get_graph(k = 5):
  graph_builder = StateGraph(State)
  llm = Llama3_2()
  # create retriever node
  embeddings = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-base")
  vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db, index_name = "typical_rag")
  retriever = vectordb.as_retriever(search_kwargs = {"k": k})
  def retrieval(state: State):
    question = state['question']
    documents = retriever.invoke(question)
    return {'question': question, 'documents': documents}
  graph_builder.add_node("retrieval", retrieval)
  # create rag node
  prompt = hub.pull("rlm/rag-prompt")
  rag_chain = prompt | llm | StrOutputParser()
  def rag(state: State):
    documents = state['documents']
    question = state['question']
    generation = rag_chain.invoke({'context': documents, 'question': question})
    return {"documents": documents, "question": question, "generation": generation}
  graph_builder.add_node("rag", rag)
  # add edges
  graph_builder.add_edge(START, "retrieval")
  graph_builder.add_edge("retrieval", "rag")
  graph_builder.add_edge("rag", END)
  graph = graph_builder.compile()
  return graph
