#!/usr/bin/python3

from absl import flags, app
import tensorrt as trt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to onnx')
  flags.DEFINE_string('output', default = 'output.engine', help = 'path to tensorrt engine')

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  runtime = trt.Runtime(TRT_LOGGER)
  with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = 1 << 30  # 1GB
    builder.max_batch_size = 32

    if fp16_mode:
      builder.fp16_mode = True

    # 解析 ONNX 模型
    with open(onnx_file_path, 'rb') as model:
      parser.parse(model.read())

    # 构建 TensorRT 引擎
    engine = builder.build_cuda_engine(network)

    # 保存 TensorRT 引擎
    with open(engine_file_path, 'wb') as f:
      f.write(engine.serialize())

def main(unused_argv):
  build_engine(FLAGS.input, FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)

