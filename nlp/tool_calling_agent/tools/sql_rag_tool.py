#!/usr/bin/python3

from typing import Union, Type, Optional
from pydantic import BaseModel, Field
from langchain import SQLDatabase
from langchain_core.runnables import Runnable
from langchain.tools import StructuredTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from .sql_rag import (
  sql_rag_chain,
  tool_name,
  tool_description,
  input_description,
  output_description
)

def load_sql_rag(llm, db):
  class SQLRAGInput(BaseModel):
    query: str = Field(description = input_description)
  class SQLRAGOutput(BaseModel):
    answer: str = Field(description = output_description)
  class SQLRAGConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    chain: Runnable
  class SQLRAGTool(StructuredTool):
    name: str = tool_name
    description: str = tool_description
    args_schema: Type[BaseModel] = SQLRAGInput
    config: SQLRAGConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> SQLRAGOutput:
      response = self.config.chain.run(query)
      return SQLRAGOutput(answer = response)
  chain = sql_rag_chain(llm, db)
  return SQLRAGTool(config = SQLRAGConfig(chain = chain))

if __name__ == "__main__":
  pass
