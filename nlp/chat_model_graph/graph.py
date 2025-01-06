#!/usr/bin/python3

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from models import Llama3_2

class Prompt(TypedDict):
  input: str
  chat_history: list

class State(TypedDict):
  messages: Annotated[list, add_messages]

def get_graph():
  graph_builder = StateGraph(State)
  llm = Llama3_2()
  def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
  graph_builder.add_node("chat", chatbot)
  graph_builder.add_edge(START,"chat")
  graph_builder.add_edge("chat", END)
  graph = graph_builder.compile()
  return graph

