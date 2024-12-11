#!/usr/bin/python3

from absl import flags, app
import torch
import torchvision

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'checkpoint.pth', help = 'path to checkpoint')
  flags.DEFINE_string('output', default = 'converted.pt', help = 'path to output torchscript model')
  flags.DEFINE_enum('model', default = 'retinanet_resnet50_fpn', enum_values = {'fasterrcnn_resnet50_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fcos_resnet50_fpn', 'retinanet_resnet50_fpn', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'maskrcnn_resnet50_fpn', 'keypointrcnn_resnet50_fpn'}, help = 'detection model')
  flags.DEFINE_enum('weights_backbone', default = 'ResNet50_Weights.IMAGENET1K_V1', enum_values = {'ResNet50_Weights.IMAGENET1K_V1', 'VGG16_Weights.IMAGENET1K_FEATURES', 'MobileNet_V3_Large_Weights.IMAGENET1K_V1'}, help = 'backbone weights')
  flags.DEFINE_integer('classnum', default = 4, help = 'class number')

def main(unused_argv):
  model = torchvision.models.get_model(FLAGS.model, weights_backbone = FLAGS.weights_backbone, num_classes = FLAGS.classnum)
  ckpt = torch.load(FLAGS.ckpt, map_location = 'cpu')
  model.load_state_dict(ckpt['model'])
  model.eval()
  example_input = torch.randn(1,3,600,800).to(torch.float32)
  scripted_model = torch.jit.trace(model, example_input)
  scripted_model.save(FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)

