#!/usr/bin/python3

from os import environ
from collections.abc import Sequence
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.messages import convert_to_messages
from prompts import HFChatPromptValue

class Llama3_2(HuggingFaceEndpoint):
  def __init__(self,):
    super(HuggingFaceEndpoint, self).__init__(
      endpoint_url = "meta-llama/Llama-3.2-3B-Instruct",
      huggingfacehub_api_token = "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",
      task = "text-generation",
      max_length = 131072,
      do_sample = False,
      top_p = 0.8,
      temperature = 0.8,
      trust_remote_code = True,
      use_cache = True
    )
  def _convert_input(self, input):
    assert isinstance(input, Sequence)
    return HFChatPromptValue(messages = convert_to_messages(input),
                             tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'))

