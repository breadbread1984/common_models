#!/usr/bin/python3

from os import environ
from transformers import AutoTokenizer
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint

class Llama3_2(ChatHuggingFace):
  def __init__(self,):
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    super(ChatHuggingFace, self).__init__(
      llm = HuggingFaceEndpoint(
        endpoint_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token = "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",
        task = "text-generation",
        max_length = 131072,
        do_sample = False,
        top_p = 0.8,
        temperature = 0.8,
        trust_remote_code = True,
        use_cache = True
      ),
      tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'),
      verbose = True
    )

