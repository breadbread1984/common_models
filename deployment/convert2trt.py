#!/usr/bin/python3

from absl import flags, app
import tensorrt as trt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to onnx')
  flags.DEFINE_string('output', default = 'output.engine', help = 'path to tensorrt engine')
  flags.DEFINE_integer('max_batch_size', default = 1, help = 'max batch size, set 0 if model has no batch dimension,')

def build_engine(onnx_file_path, engine_file_path, max_batch_size = 1, fp16_mode = True):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(TRT_LOGGER)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, TRT_LOGGER)

  with open(FLAGS.input, 'rb') as f:
    if not parser.parse(f.read()):
      print(f"Failed to parse the ONNX file: {FLAGS.input}")
      for error in range(parser.num_errors):
        print(parser.get_error(error))
      return None

  config = builder.create_builder_config()
  config.max_workspace_size = 1 << 30  # 设置最大工作空间大小（1GB）
  if fp16_mode:
    config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16 模式
  builder.max_batch_size = max_batch_size
  print("Building TensorRT engine. This may take a few minutes...")
  engine = builder.build_engine(network, config)
  if engine is None:
    print("Failed to create the TensorRT engine.")
    return None

  with open(engine_file_path, "wb") as f:
    f.write(engine.serialize())
  print(f"Engine has been saved to {engine_file_path}")

  return engine

def main(unused_argv):
  build_engine(FLAGS.input, FLAGS.output, FLAGS.max_batch_size)

if __name__ == "__main__":
  add_options()
  app.run(main)

