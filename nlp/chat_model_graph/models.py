#!/usr/bin/python3

from os import environ
from langchain_community.llms import HuggingFaceEndpoint

def LLM():
  environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
  return HuggingFaceEndpoint(
    endpoint_url = "meta-llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    max_length = 131072,
    do_sample = False,
    top_p = 0.8,
    temperature = 0.8,
    trust_remote_code = True,
    use_cache = True
  )

