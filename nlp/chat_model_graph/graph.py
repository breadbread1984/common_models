#!/usr/bin/python3

from typing import Annotated
from tying_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from models import Qwen2
from prompts import get_prompt

class State(TypedDict):
  messages: Annotated[list, add_messages]

def get_graph():
  graph_builder = StateGraph(State)
  template = get_prompt()
  llm = Qwen2()
  def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
  graph_builder.add_node("prompt", template)
  graph_builder.add_node("chat", chatbot)
  graph_builder.add_edge(START,"prompt")
  graph_builder.add_edge("prompt", "chat")
  graph_builder.add_edge("chat", END)
  graph = graph_builder.compile()
  return graph

