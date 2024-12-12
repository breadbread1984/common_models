#!/usr/bin/python3

from absl import flags, app
import tritonclient.http as httpclient
import numpy as np
import cv2
import torch
import torchvision # for nms op availability in torchscript

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = "localhost", help = 'host')
  flags.DEFINE_integer('port', default = 8081, help = 'port')
  flags.DEFINE_string('input', default = 'pics/000035.jpg', help = 'path to picture')
  flags.DEFINE_enum('method', default = 'local', enum_values = {'local', 'network'}, help = 'which method to use')
  flags.DEFINE_string('model', default = 'models/torch_model/1/model.pt', help = 'path to model')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  img = cv2.imread(FLAGS.input)
  inputs = img[:,:,::-1]
  inputs = np.ascontiguousarray(inputs)
  inputs = (inputs / 255).astype(np.float32)
  inputs = np.transpose(inputs, (2,0,1))
  if FLAGS.method == 'network':
    client = httpclient.InferenceServerClient(f"{FLAGS.host}:{FLAGS.port}")
    feeds = [httpclient.InferInput("inputs", inputs.shape, "FP32")]
    feeds[0].set_data_from_numpy(inputs)
    outputs = [httpclient.InferRequestedOutput("boxes"),
               httpclient.InferRequestedOutput("scores"),
               httpclient.InferRequestedOutput("labels")]
    response = client.infer("torch_model", inputs = feeds, outputs = outputs, model_version = "1")
    boxes, scores, labels = response.get_output("boxes"), response.get_output("scores"), response.get_output("labels")
  elif FLAGS.method == 'local':
    model = torch.jit.load(FLAGS.model, map_location = FLAGS.device)
    boxes, scores, labels = model.forward(torch.from_numpy(inputs).to(FLAGS.device))
    boxes, scores, labels = boxes.detach().cpu().numpy(), scores.detach().cpu().numpy(), labels.detach().cpu().numpy()
  else:
    raise Exception('error method')
  # visualize
  for box, score, label in zip(boxes, scores, labels):
    if score < 0.5: continue
    color = (255 if label == 0 else 0,255 if label == 1 else 0,255 if label == 2 else 0)
    cv2.rectangle(img, tuple(box[:2].astype(np.int32).tolist()), tuple(box[2:].astype(np.int32).tolist()), color, 2, 1)
  cv2.imwrite('output.png', img)
  cv2.imshow('', img)
  cv2.waitKey()

if __name__ == "__main__":
  add_options()
  app.run(main)

