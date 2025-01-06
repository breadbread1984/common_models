#!/usr/bin/python3

from absl import flags, app
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from graph import get_graph
from prompts import get_prompt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'service host')
  flags.DEFINE_integer('port', default = 8081, help = 'service port')

def create_interface():
  graph = get_graph()
  prompt = get_prompt()
  def chatbot_response(user_input, history):
    chat_history = list()
    for human, ai in history:
      chat_history.append(HumanMessage(content = human))
      chat_history.append(AIMessage(content = ai))
    messages = prompt.invoke({'input': user_input, 'chat_history': chat_history}).to_messages()
    for event in graph.stream({"messages": messages}):
      for value in event.values():
        response = value["messages"][-1].content
        break
    history.append((user_input, response))
    return history, history, ""
  with gr.Blocks() as demo:
    state = gr.State([])
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>对话系统</center></h1>")
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
  demo = create_interface()
  demo.launch(server_name = FLAGS.host,
              server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)
