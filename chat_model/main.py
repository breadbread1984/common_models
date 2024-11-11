#!/usr/bin/python3

from absl import flags, app
import gradio as gr
from chains import chat_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('tgi_host', default = 'http://localhost:8080/generate', help = 'tgi url')
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'service host')
  flags.DEFINE_integer('port', default = 8081, help = 'service port')

def create_interface():
  chain = chat_chain(FLAGS.tgi_host)
  def chatbot_response(user_input, history):
    response = chain.invoke({'input': user_input})
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
