#!/usr/bin/python3

from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

def rag_chain(llm, neo4j_host, neo4j_user, neo4j_password, neo4j_db):
  embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db, index_name = "typical_rag")
  retriever = vectordb.as_retriever()
  # chain to summarize chat history into a standalone question
  history_aware_retriever = create_history_aware_retriever(llm, retriever, hub.pull("langchain-ai/chat-langchain-rephrase"))
  if True:
    # swith on this branch if want to watch retrieval results
    def print_results(x):
      print(x)
      return x
    history_aware_retriever = history_aware_retriever | print_results
  # chain to answer the question based on retrievaled context
  combine_docs_chain = create_stuff_documents_chain(llm, hub.pull("langchain-ai/retrieval-qa-chat"))
  chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
  return chain

if __name__ == "__main__":
  pass
