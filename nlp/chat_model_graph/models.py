#!/usr/bin/python3

from langchain_community.llms import HuggingFaceEndpoint

def Qwen2():
  return HuggingFaceEndpoint(
    endpoint_url = "Qwen/Qwen2.5-7B-Instruct",
    task = "text-generation",
    max_length = 131072,
    do_sample = False,
    top_p = 0.8,
    temperature = 0.8,
    trust_remote_code = True,
    use_cache = True
  )

