#!/usr/bin/python3

from typing import Union, Type, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain.tools import StructuredTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from .graph_rag import (
  input_description,
  output_description,
  tool_name,
  tool_description,
  graph_rag_chain,
  neo4j_host,
  neo4j_user,
  neo4j_password,
  neo4j_db
)

def load_graph_rag(llm):
  class GraphRAGInput(BaseModel):
    query: str = Field(description = input_description)
  class GraphRAGOutput(BaseModel):
    answer: str = Field(description = output_description)
  class GraphRAGConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    chain: Runnable
  class GraphRAGTool(StructuredTool):
    name: str = tool_name
    description: str = tool_description
    args_schema: Type[BaseModel] = GraphRAGInput
    config: GraphRAGConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> GraphRAGOutput:
      response = self.config.chain.invoke({'query': query, 'chat_history': []})
      return RAGOutput(answer = response['result'])
  chain = graph_rag_chain(llm, neo4j_host, neo4j_user, neo4j_password, neo4j_db)
  return GraphRAGTool(config = GraphRAGConfig(chain = chain))

if __name__ == "__main__":
  pass
