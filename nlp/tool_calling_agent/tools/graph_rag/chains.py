#!/usr/bin/python3

from langchain import hub
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from .prompts import qa_prompt, cypher_prompt, fewshot_cypher_prompt

def graph_rag_chain(llm, neo4j_host = 'bolt://localhost:7687', neo4j_user = 'neo4j', neo4j_password = 'neo4j', neo4j_db = 'neo4j', use_fewshot = False, use_selector = False):
  graph = Neo4jGraph(url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db)
  graph.refresh_schema()
  print(graph.schema)
  chain = GraphCypherQAChain.from_llm(
    graph = graph,
    llm = llm,
    qa_prompt = qa_prompt(),
    cypher_prompt = cypher_prompt() if not use_fewshot else \
                    fewshot_cypher_prompt(use_selector, neo4j_host, neo4j_user, neo4j_password, neo4j_db),
    verbose = True,
    allow_dangerous_requests = True
  )
  return chain
