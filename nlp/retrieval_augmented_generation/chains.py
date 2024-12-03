#!/usr/bin/python3

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain.schema.runnable import ConfigurableField
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from prompts import condense_question_prompt

def condense_chain(tokenizer, llm, neo4j_host, neo4j_user, neo4j_password, neo4j_db):
  embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db, index_name = "typical_rag")
  parent_vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db,
    retrieval_query = """
    MATCH (node)<-[:HAS_CHILD]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata LIMIT 1
    """,
    index_name = "parent_document"
  )
  hypothetic_question_vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db,
    retrieval_query = """
    MATCH (node)<-[:HAS_QUESTION]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """,
    index_name = "hypothetic_question_query"
  )
  summary_vectordb = Neo4jVector(embedding = embedding, url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db,
    retrieval_query = """
    MATCH (node)<-[:HAS_SUMMARY]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """,
    index_name = "summary"
  )
  chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever = vectordb.as_retriever().configurable_alternatives(
      ConfigurableField(id = "strategy"),
      default_key = "typical_rag",
      parent_strategy = parent_vectordb.as_retriever(),
      hypothetical_questions = hypothetic_question_vectordb.as_retriever(),
      summary_strategy = summary_vectordb.as_retriever()
    ),
    condense_question_prompt = condense_question_prompt(tokenizer),
  )
  return chain

if __name__ == "__main__":
  pass  
