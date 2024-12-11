#!/usr/bin/python3

from os import makedirs
from os.path import dirname, exists
from absl import flags, app
import torch
from torch import nn
import torchvision

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'checkpoint.pth', help = 'path to checkpoint')
  flags.DEFINE_string('output', default = 'models/torch_model/1/model.pt', help = 'path to output torchscript model')
  flags.DEFINE_enum('model', default = 'retinanet_resnet50_fpn', enum_values = {'fasterrcnn_resnet50_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fcos_resnet50_fpn', 'retinanet_resnet50_fpn', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'maskrcnn_resnet50_fpn', 'keypointrcnn_resnet50_fpn'}, help = 'detection model')
  flags.DEFINE_enum('weights_backbone', default = 'ResNet50_Weights.IMAGENET1K_V1', enum_values = {'ResNet50_Weights.IMAGENET1K_V1', 'VGG16_Weights.IMAGENET1K_FEATURES', 'MobileNet_V3_Large_Weights.IMAGENET1K_V1'}, help = 'backbone weights')
  flags.DEFINE_integer('classnum', default = 4, help = 'class number')
  flags.DEFINE_enum('type', default = 'trace', enum_values = {'trace', 'script'}, help = 'which way to generate torchscript')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'which device to use')

class Wrapper(nn.Module):
  def __init__(self, model, backbone, num_classes, ckpt):
    super(Wrapper, self).__init__()
    ckpt = torch.load(FLAGS.ckpt, map_location = FLAGS.device)
    self.model = torchvision.models.get_model(model, weights_backbone = backbone, num_classes = num_classes).to(FLAGS.device)
    self.model.load_state_dict(ckpt['model'])
  def forward(self, inputs):
    detections = self.model([inputs])
    detection = detections[0]
    return detection['boxes'], detection['scores'], detection['labels']

def main(unused_argv):
  makedirs(dirname(FLAGS.output), exist_ok = True)
  model = Wrapper(FLAGS.model, backbone = FLAGS.weights_backbone, num_classes = FLAGS.classnum, ckpt = FLAGS.ckpt).to(FLAGS.device)
  model.eval()
  if FLAGS.type == 'trace':
    example_input = torch.randn(3,600,800).to(torch.float32).to(FLAGS.device)
    trace_model = torch.jit.trace(model, example_input)
    trace_model.save(FLAGS.output)
  elif FLAGS.type == 'script':
    script_model = torch.jit.script(model)
    script_model.save(FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)

