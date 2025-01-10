#!/usr/bin/python3

from langchain import hub
from langchain import SQLDatabase
from langchain.agents import AgentExecutor, create_tool_calling_agent
from models import Llama3_2, CodeLlama, Qwen2_5, CodeQwen2
from tools import load_graph_rag, load_rag, load_sql_rag

class Agent(object):
  def __init__(self, model = 'llama3', tools = ['google-serper', 'llm-math', 'wikipedia', 'arxiv'], **kwargs):
    llms_types = {
      'llama3': Llama3_2,
      'codellama': CodeLlama,
      'qwen2': Qwen2_5,
      'codeqwen': CodeQwen2
    }
    llm = llms_types[model]()
    db = SQLDatabase.from_uri(f"sqlite:///{kwargs.get('sqlite_path')}")
    tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663') + \
            [
              load_graph_rag(llm),
              load_sql_rag(llm, db),
              load_rag(llm)
            ]
    prompt = hub.pull('hwchase17/openai-functions-agent')
    agent = create_tool_calling_agent(chat_model, tools, prompt)
    self.agent_chain = AgentExecutor(agent = agent, tools = tools, verbose = True)
  def query(self, question):
    return self.agent_chain.invoke({"input": question})

if __name__ == "__main__":
  agent = Agent(sqlite_path = "test/Chinook_Sqlite.sqlite")
  print(agent.query("Which organization does Elon Musk work as CEO of?"))
