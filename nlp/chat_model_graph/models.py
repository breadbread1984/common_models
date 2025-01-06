#!/usr/bin/python3

from os import environ
from collections.abc import Sequence
from transformers import AutoTokenizer
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import convert_to_messages

class Llama3_2(ChatHuggingFace):
  def __init__(self,):
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    super(ChatHuggingFace, self).__init__(
      llm = HuggingFaceEndpoint(
        repo_id = "meta-llama/Llama-3.2-3B-Instruct",
        task = "text-generation",
        max_tokens = 131072,
        do_sample = False,
        top_p = 0.8,
        temperature = 0.8,
        trust_remote_code = True,
        use_cache = True
      ),
      verbose = True
    )

