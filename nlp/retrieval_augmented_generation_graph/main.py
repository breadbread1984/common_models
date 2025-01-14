#!/usr/bin/python3

from absl import flags, app
import pandas as pd
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from graph import get_graph

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'service host')
  flags.DEFINE_integer('port', default = 8081, help = 'service port')
  flags.DEFINE_integer('rank', default = 3, help = 'rank')

def create_interface():
  graph = get_graph()
  def chatbot_response(user_input, history):
    chat_history = list()
    for human, ai in history:
      chat_history.append(HumanMessage(content = human))
      chat_history.append(AIMessage(content = ai))
    for event in graph.stream({"question": user_input, 'rank': FLAGS.rank}):
      if 'rag' not in event: continue
      response = event['rag']['generation']
      documents = event['rag']['documents']
      break
    history.append((user_input, response))
    ids = [i for i in range(len(documents))]
    contents = [doc.page_content for doc in documents]
    urls = [doc.metadata['url'] for doc in documents]
    classifications = [doc.metadata['classification'] for doc in documents]
    documents = pd.DataFrame({'id': ids, 'content': contents, 'url': urls, 'classification': classifications})
    return history, history, "", documents
  with gr.Blocks() as demo:
    state = gr.State([])
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>对话系统</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        references = gr.Dataframe(value = pd.DataFrame({'id':[], 'content':[], 'url': [], 'classification': []}), label = 'references')
        user_input = gr.Textbox(label = '需要问什么？')
        with gr.Row():
          submit_btn = gr.Button("发送")
        with gr.Row():
          clear_btn = gr.ClearButton(components = [chatbot, state], value = "清空问题")
      submit_btn.click(chatbot_response,
                       inputs = [user_input, state],
                       outputs = [chatbot, state, user_input, references])
  return demo

def main(unused_argv):
  demo = create_interface()
  demo.launch(server_name = FLAGS.host,
              server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)
