#!/usr/bin/python3

import json
import requests
from pydantic import BaseModel
from operator import itemgetter
from langchain.llms.base import LLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.utils.pydantic import is_basemodel_instance, is_basemodel_subclass
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables.base import RunnableLambda, RunnableMap
from langchain_core.runnables.passthrough import RunnablePassthrough

class StructuredLLM(LLM):
  def parse_response(self, message) -> str:
    """Extract `function_call` from `AIMessage`."""
    if isinstance(message, AIMessage):
      kwargs = message.additional_kwargs
      tool_calls = message.tool_calls
      if len(tool_calls) > 0:
        tool_call = tool_calls[-1]
        args = tool_call.get("args")
        return json.dumps(args)
      elif "function_call" in kwargs:
        if "arguments" in kwargs["function_call"]:
          return kwargs["function_call"]["arguments"]
        raise ValueError(
          f"`arguments` missing from `function_call` within AIMessage: {message}"
        )
      else:
        raise ValueError("`tool_calls` missing from AIMessage: {message}")
    raise ValueError(f"`message` is not an instance of `AIMessage`: {message}")

  def _is_pydantic_class(self, obj) -> bool:
    return isinstance(obj, type) and (
      is_basemodel_subclass(obj) or BaseModel in obj.__bases__
    )
  def with_structured_output(self, schema, *, include_raw = False, **kwargs):
    if kwargs:
      raise ValueError(f"Received unsupported arguments {kwargs}")
    is_pydantic_schema = self._is_pydantic_class(schema)
    if schema is None:
      raise ValueError(
        "schema must be specified when method is 'function_calling'. "
        "Received None."
      )
    llm = self.bind(tools=[schema], format="json") # give tools argument to llm.invoke. returned llm type is RunnableBindingBase.
    if is_pydantic_schema:
      output_parser: OutputParserLike = PydanticOutputParser(  # type: ignore[type-var]
        pydantic_object=schema  # type: ignore[arg-type]
      )
    else:
      output_parser = JsonOutputParser()

    parser_chain = RunnableLambda(self.parse_response) | output_parser
    if include_raw:
      parser_assign = RunnablePassthrough.assign(
        parsed=itemgetter("raw") | parser_chain, parsing_error=lambda _: None
      )
      parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
      parser_with_fallback = parser_assign.with_fallbacks(
        [parser_none], exception_key="parsing_error"
      )
      return RunnableMap(raw=llm) | parser_with_fallback
    else:
      return llm | parser_chain

class TGI(StructuredLLM):
  url: str = None
  headers: dict = None
  def __init__(self, host):
    super(TGI, self).__init__()
    self.url = host
    self.headers = {'Authorization': "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"}
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    data = {"inputs": prompt, "parameters": {"temperature": 0.6, "top_p": 0.9}}
    for i in range(10):
      response = requests.post(self.url, headers = self.headers, json = data)
      if response.status_code == 200:
        break
    else:
      raise Exception(f'请求失败{response.status_code}')
    return response.json()['generated_text']
  @property
  def _llm_type(self):
    return "tgi"

if __name__ == "__main__":
  tgi = TGI('http://localhost:8080/generate')
