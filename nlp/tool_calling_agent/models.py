#!/usr/bin/python3

from os import environ
from langchain import hub
from transformers import AutoTokenizer
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough

class Llama3_2(ChatHuggingFace):
  def __init__(self,):
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    super(ChatHuggingFace, self).__init__(
      llm = HuggingFaceEndpoint(
        endpoint_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token = "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",
        task = "text-generation",
        #max_length = 131072,
        do_sample = False,
        top_p = 0.8,
        temperature = 0.8,
        #use_cache = True
      ),
      tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'),
      verbose = True
    )
  def _generate(
        self,
        messages,
        stop = None,
        run_manager = None,
        **kwargs,
    ):
    if 'tools' not in kwargs:
      # ordinary LLM inference
      llm_input = self._to_chat_prompt(messages)
      llm_result = self.llm._generate(
        prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
      )
      return self._to_chat_result(llm_result)
    else:
      # create agent to generate tool calls
      raise NotImplementedError('huggingface does not support tool calling')

class Qwen2_5(ChatHuggingFace):
  def __init__(self,):
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    super(ChatHuggingFace, self).__init__(
      llm = HuggingFaceEndpoint(
        endpoint_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token = "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",
        task = "text-generation",
        do_sample = False,
        top_p = 0.8,
        temperature = 0.8
      ),
      tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'),
      verbose = True
    )
  def _generate(
        self,
        messages,
        stop = None,
        run_manager = None,
        **kwargs,
    ):
    if 'tools' not in kwargs:
      # ordinary LLM inference
      llm_input = self._to_chat_prompt(messages)
      llm_result = self.llm._generate(
        prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
      )
      return self._to_chat_result(llm_result)
    else:
      # create agent to generate tool calls
      raise NotImplementedError

if __name__ == "__main__":
  from langchain_core.tools import tool
  from langchain.agents import AgentExecutor, create_tool_calling_agent

  @tool
  def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

  @tool
  def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

  chat_model = Llama3_2()
  tools = [add, multiply]
  if False:
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": "What is 3 * 12?"})
  else:
    chat_model = chat_model.bind_tools(tools)
    response = chat_model.invoke([('user', 'What is 3 * 12?')])
    print(response.tool_calls)
  print(response)
