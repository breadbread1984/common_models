#!/usr/bin/python3

from absl import flags, app
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from chains import load_graph_qa_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('tgi_host', default = 'http://localhost:8080/generate', help = 'host of TGI')
  flags.DEFINE_string('neo4j_host', default = 'bolt://localhost:7687', help = 'host')
  flags.DEFINE_string('neo4j_user', default = 'neo4j', help = 'user name')
  flags.DEFINE_string('neo4j_password', default = 'neo4j', help = 'password')
  flags.DEFINE_string('neo4j_db', default = 'neo4j', help = 'database')
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'service host')
  flags.DEFINE_integer('port', default = 8081, help = 'service port')
  flags.DEFINE_boolean('use_fewshot', default = False, help = 'whether to use few shot')
  flags.DEFINE_boolean('use_selector', default = False, help = 'whether to use example selector')

def create_interface(chain):
  def chatbot_response(user_input, history):
    chat_history = list()
    for human, ai in history:
      chat_history.append(HumanMessage(content = human))
      chat_history.append(AIMessage(content = ai))
    response = chain.invoke({'query': user_input, 'chat_history': chat_history})
    history.append((user_input, response))
    return history, history, ""
  with gr.Blocks() as demo:
    state = gr.State([])
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>RAG系统</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        user_input = gr.Textbox(label = '需要问什么？')
        with gr.Row():
          submit_btn = gr.Button("发送")
        with gr.Row():
          clear_btn = gr.ClearButton(components = [chatbot, state], value = "清空问题")
      submit_btn.click(chatbot_response,
                       inputs = [user_input, state],
                       outputs = [chatbot, state, user_input])
  return demo

def main(unused_argv):
  chain = load_graph_qa_chain(FLAGS.tgi_host, FLAGS.neo4j_host, FLAGS.neo4j_user, FLAGS.neo4j_password, FLAGS.neo4j_db, use_fewshot = FLAGS.use_fewshot, use_selector = FLAGS.use_selector)
  demo = create_interface(chain)
  demo.launch(server_name = FLAGS.host, server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)
