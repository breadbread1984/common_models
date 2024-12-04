#!/usr/bin/python3

from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from models import TGI
from prompts import qa_prompt, cypher_prompt

def load_graph_qa_chain(tgi_host, neo4j_host, neo4j_user, neo4j_password, neo4j_db):
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  llm = TGI(tgi_host)
  graph = Neo4jGraph(url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db)
  graph.refresh_schema()
  print(graph.schema)
  chain = GraphCypherQAChain.from_llm(
    graph = graph,
    llm = llm,
    qa_prompt = qa_prompt(tokenizer),
    cypher_prompt = cypher_prompt(tokenizer),
    verbose = True,
    allow_dangerous_requests = True
  )
  return chain

