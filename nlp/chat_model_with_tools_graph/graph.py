#!/usr/bin/python3

from os import environ
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.agents import load_tools
from models import Llama3_2
from tools import BasicToolNode

class State(TypedDict):
  messages: Annotated[list, add_messages]

def get_graph():
  graph_builder = StateGraph(State)
  llm = Llama3_2()
  tools = load_tools(['google-serper', 'llm-math', 'wikipedia', 'arxiv', 'openweathermap-api'], llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663', openweathermap_api_key = '')
  # FIXME: huggingface chat model cannot support tool calling function
  llm_with_tools = llm.bind_tools(tools)
  def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
  graph_builder.add_node("chatbot", chatbot)
  environ['SERPER_API_KEY'] = 'd075ad1b698043747f232ec1f00f18ee0e7e8663'
  tool_node = BasicToolNode(tools = tools)
  graph_builder.add_node("tools", tool_node)
  graph_builder.add_edge(START, "chatbot")
  def route_tools(state: State):
    # edge switcher
    # if message container tool_calls switch tools
    # else switch to end
    if isinstance(state, list):
      ai_message = state[-1]
    elif messages := state.get("messages", []):
      ai_message = messages[-1]
    else:
      raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
      return "tools"
    return END
  graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
  graph_builder.add_edge("tools", "chatbot")
  graph = graph_builder.compile()
  return graph

