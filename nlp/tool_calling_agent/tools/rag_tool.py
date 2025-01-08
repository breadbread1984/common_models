#!/usr/bin/python3

from typing import Union, Type, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain.tools import StructuredTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from .rag import (
  input_description,
  output_description,
  tool_name,
  tool_description,
  rag_chain,
  neo4j_host,
  neo4j_user,
  neo4j_password,
  neo4j_db
)

def load_rag(llm):
  class RAGInput(BaseModel):
    query: str = Field(description = input_description)
  class RAGOutput(BaseModel):
    answer: str = Field(description = output_description)
  class RAGConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    chain: Runnable
  class RAGTool(StructuredTool):
    name: str = tool_name
    description: str = tool_description
    args_schema: Type[BaseModel] = RAGInput
    config: RAGConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> RAGOutput:
      response = self.config.invoke({'input': query, 'chat_history': []})
      return RAGOutput(answer = response['answer'])
  chain = rag_chain(llm, neo4j_host, neo4j_user, neo4j_password, neo4j_db)
  return RAGTool(config = RAGConfig(chain = chain))

if __name__ == "__main__":
  tool = load_rag()
  res = tool.invoke({"query": ""})
  print(res)
