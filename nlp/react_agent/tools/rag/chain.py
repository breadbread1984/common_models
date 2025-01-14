#!/usr/bin/python3

from os import environ
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

def rag_chain(llm, neo4j_host, neo4j_user, neo4j_password, neo4j_db, k = 7):
  embedding = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-base")
  vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db, index_name = "typical_rag")
  retriever = vectordb.as_retriever(search_kwargs = {"k": k})
  # chain to summarize chat history into a standalone question
  history_aware_retriever = create_history_aware_retriever(llm, retriever, hub.pull("langchain-ai/chat-langchain-rephrase"))
  # chain to answer the question based on retrievaled context
  combine_docs_chain = create_stuff_documents_chain(llm, hub.pull("langchain-ai/retrieval-qa-chat"))
  chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
  return chain

if __name__ == "__main__":
  pass
