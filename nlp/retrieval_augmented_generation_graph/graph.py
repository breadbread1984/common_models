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

def get_graph():
  graph_builder = StateGraph(State)
  llm = Llama3_2()
  # create rephrase question node
  re_write_prompt = ChatPromptTemplate.from_messages([
    ('system', """You a question re-writer that converts an input question to a better version that is optimized \n 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
    ('user', "Here is the initial question: \n\n {question} \n Formulate an improved question.")
  ])
  question_rewriter = re_write_prompt | llm | StrOutputParser()
  def transform_query(state: State):
    question = state['question']
    documents = state['documents']
    rephrased_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": rephrased_question}
  graph_builder.add_node("transform", transform_query)
  # create retriever node
  embeddings = HuggingFaceEmbeddings(model = "intfloat/multilingual-e5-base")
  vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db, index_name = "typical_rag")
  retriever = vectordb.as_retriever() # search_kwargs = {"k": 5}
  def retrieval(state: State):
    question = state['question']
    documents = retriever.invoke(question)
    return {'question': question, 'documents': documents}
  graph_builder.add_node("retrieval", retrieval)
  # create grade documents node
  grade_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a grader assessing relevance of a retrieved document to a user question. \n 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""),
    ("user", "Retrieved document: \n\n {document} \n\n User question: {question}")
  ])
  class GradeDocuments(BaseModel):
    binary_score: str = Field(description = "Documents are relevent to the question, 'yes' or 'no'")
  structured_llm_grader = llm.with_structured_output(GradeDocuments)
  grade_chain = grade_prompt | structured_llm_grader
  def grade_documents(state: State):
    question = state['question']
    documents = state['documents']
    filtered_docs = list()
    for d in documents:
      score = retrieval_grader.invoke({'question': question, 'document': d.page_content})
      grade = score.binaryd_score
      if grade = "yes":
        filtered_docs.append(d)
      else:
        continue
    return {"documents": filtered_docs, "question": question}
  graph_builder.add_node("grader", grade_documents)
  # create chain node
  prompt = hub.pull("rlm/rag-prompt")
  rag_chain = prompt | llm | StrOutputParser()
  def chatbot(state: State):
    documents = state['documents']
    question = state['question']
    generation = rag_chain.invoke({'context': documents, 'question': question})
    return {"documents": documents, "question": question, "generation": generation}
  graph_builder.add_node("chatbot", chatbot)
  # add edges
  graph_builder.add_edge(START, "retrieval")
  graph_builder.add_edge("retrieval", "grader")
  def whether_to_rephrase(state: State):
    filtered_documents = state['documents']
    if len(filtered_documents) == 0:
      return "transform"
    else:
      return "chatbot"
  graph_builder.add_conditional_edges("grader", whether_to_rephrase, {"transform": "transform", "chatbot": "chatbot"})
  graph_builder.add_conditional_edges("transform", "retrieval")
  class GradeHallucinations(BaseModel):
    binary_score: str = Field(description = "Answer is grounded in the facts, 'yes' or 'no'")
  hallucination_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
    ('user', "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
  ])
  structured_llm_grader = llm.with_structured_output(GradeHallucinations)
  hallucination_grader = hallucination_prompt | structured_llm_grader
  class GradeAnswer(BaseModel):
    binary_score: str = Field(description = "Answer addresses the question, 'yes' or 'no'")
  answer_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""),
    ('user', "User question: \n\n {question} \n\n LLM generation: {generation}")
  ])
  structured_llm_grader = llm.with_structured_output(GradeAnswer)
  answer_grader = answer_prompt | structured_llm_grader
  def answer_grader(state: State):
    documents = state['documents']
    generation = state['generation']
    question = state['question']
    score = hallucination_grader.invoke({'documents': documents, 'generation': generation})
    if score.binary_score == 'yes':
      score = answer_grader.invoke({"question": question, "generation": generation})
      if score.binary_score == 'yes':
        return "useful"
      else:
        return "not useful"
    else:
      return "not supported"
  graph_builder.add_conditional_edges("chatbot", answer_grader, {"not supported": "chatbot", "useful": END, "not useful": "transform"})
  graph = graph_builder.compile()
  return graph
