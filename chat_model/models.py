#!/usr/bin/python3

import requests
from langchain.llms.base import LLM

def TGI(host = 'http://localhost:8080/generate'):
  class TGILLM(LLM):
    url: str = None
    headers: dict = None
    def __init__(self, host):
      super().__init__()
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
  llm = TGILLM(host)
  return llm

