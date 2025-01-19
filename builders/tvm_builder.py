from os import listdir, remove, path, makedirs
from os.path import isfile, isdir, join, exists, normpath, basename
import onnx
import tvm
import tvm.relay as relay
from generators import model as model_generator

from pathlib import Path

class TVMBuilder:

    def __init__(self, config):
        self.build = config["build"]

    def build_tvm(self, onnx_path, output_model_path, input_shape):
        onnx_model = onnx.load(onnx_path)
        input_name = onnx_model.graph.input[0].name
        # Input shape is preserved across models.
        shape_dict = {input_name: input_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        target = tvm.target.Target(self.build["target"], host=self.build["host"])
        file_name = Path(onnx_path).stem
        
        file_path = model_generator.generate_original_model(mod, target, params, output_model_path, file_name=file_name, opt_level=2)
        return file_path

    def get_layers_of_type(self, nodes, op_types):
        return [i for i in range(len([node for node in nodes if node.op_type in op_types]))]