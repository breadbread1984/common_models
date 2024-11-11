#!/usr/bin/python3

from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate

class HFChatPromptTemplate(ChatPromptTemplate):
  def __init__(self,
               messages,
               model_id,
               *,
               template_format = "f-string",
               **kwargs):
    super(HFChatPromptTemplate, self).__init__(messages, template_format = template_format, **kwargs)
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
  def format_prompt(self, **kwargs):
    messages = self.format_message(**kwargs)
    string_messages = []
    for m in messages:
      if isinstance(m, HumanMessage):
        role = 'user'
      elif isinstance(m, AIMessage):
        role = 'assistant'
      elif isinstance(m, SystemMessage):
        role = 'system'
      else:
        raise Exception(f'Got unsupported message type: {m}')
    messages.append({'role': role, 'content': m.content})
    return self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

if __name__ == "__main__":
  prompt = HFChatPromptTemplate([
    ('system', "you are a helpful AI bot. Your name is {name}"),
    ('human', '{user_input}')
  ], 'Qwen/Qwen2.5-7B-Instruct')
  txt = prompt.invoke({'user_input': 'what is your name', 'name': 'robot test'})
  print(txt)
