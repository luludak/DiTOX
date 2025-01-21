import onnx
import onnxruntime as ort
import numpy as np
from os import listdir, remove, path, makedirs
from os.path import isfile, isdir, join, exists, normpath, basename

from scipy.special import softmax

from processors.model_preprocessor import ModelPreprocessor
from evaluators.evalgenerator import EvaluationGenerator

from pathlib import Path
from PIL import Image

class ONNXRunner:

    def __init__(self, config, onnx_model, topK=10):
        self.evaluation_generator = EvaluationGenerator()
        self.topK = topK
        self.onnx_model = onnx_model
        self.ort_sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])

    def evaluate(self, source_run_obj, target_run_obj, type=["classification"]):
        return self.evaluation_generator.generate_objects_comparison(source_run_obj, target_run_obj, type)

    def execute_and_evaluate_single_model(self, onnx_model, run_obj, image_path, config, include_certainties=False):
        image_obj = self.execute_onnx_model(onnx_model, [image_path], config, print_percentage=False, include_certainties=include_certainties)
        image_name = list(image_obj.keys())[0]
        return self.evaluate(run_obj, image_obj)["images"][image_name]

    def execute_and_evaluate_single(self, onnx_path, run_obj, image_path, config, include_certainties=False):
        image_obj = self.execute_onnx_path(onnx_path, [image_path], config, print_percentage=False, include_certainties=include_certainties)
        image_name = list(image_obj.keys())[0]
        return self.evaluate(run_obj, image_obj)["images"][image_name]

    def evaluate_single(self, run_obj, image_obj):
        # Return single evaluation.
        image_name = list(image_obj.keys())[0]
        return self.evaluate(run_obj, image_obj)["images"][image_name]        

    def get_nodes_containing_input(self, base_model, input_name):

        base_nodes = base_model.graph.node
        nodes_found = []
        for base_node in base_nodes:
            if input_name in base_node.input:
                nodes_found.append(base_node)
        return nodes_found


    def execute_onnx_model(self, images_paths, config, image_names=None, print_percentage=True, include_certainties=False):
        # Execute and return all data.
        
        has_symbolic_input = False
        input_dimension = config["input_dimension"]
        if len(input_dimension) == 0:
            input_dimension = [224, 224]
            has_symbolic_input = True
        
        self.preprocessing_data = {
            "input": config["input_shape"],
            "input_dimension": input_dimension,
            "library": None
        }

        # Best effort approach for symbolic dimensions.

        # Set library to None, so that the name is utilized for preprocessing selection.
        model_preprocessor = ModelPreprocessor(self.preprocessing_data)

        output_data = {}
        count = 0
        images_length = len(images_paths)
        step = (images_length // 4) if images_length > 4 else images_length

        for img_path in images_paths:
            
            # if(print_percentage and count % step == 0):
            #     print("Complete: " + str((count // step) * 25) + "%")

            count += 1
            img_name = Path(img_path).name
            img = Image.open(img_path)

            # Convert monochrome to RGB.
            if (len(config["input_shape"]) > 3 and \
                (config["input_shape"][1] == 1 or config["input_shape"][3] == 1)):
                img = img.convert("L")
                # print(np.array(img).shape)
            else:
                if(img.mode == "L" or img.mode == "BGR"):
                    img = img.convert("RGB")
                

            img = img.resize(input_dimension)
            img = model_preprocessor.preprocess(config["model_name"], img, True)

            input_name = self.onnx_model.graph.input[0].name if "input_name" not in config else config["input_name"]
            # print(img.astype(np.float32))
            shape = self.onnx_model.graph.input[0].type.tensor_type.shape.dim
            if (len(shape) < len(img.shape)):
                img = np.squeeze(img)
            
            # TODO: Refactor. Hack for SSD-MobilenetV1 models.
            input_obj = {input_name: img.astype(np.uint8 if has_symbolic_input else np.float32)}
            if len(self.onnx_model.graph.input) == 2:
                input_obj["image_shape"] = [[224.0, 224.0]]
            # print(input_obj)
            output = self.ort_sess.run(None, input_obj)

            if len(output) == 1 and np.array(output).ndim == 1:

                scores = softmax(output)

                scores = np.squeeze(scores)
                ranks = np.argsort(scores)[::-1]
                extracted_ranks = ranks[0:self.topK]
                if include_certainties:
                    output_data[img_name] = [(rank, str(scores[rank])) for rank in extracted_ranks.tolist()]
                else:
                    output_data[img_name] = extracted_ranks.tolist()
            else:
                if (img_name not in output_data):
                    output_data[img_name] = []

                for i, output_tensor in enumerate(output):
                    output_tensor = np.squeeze(output_tensor)
                    if (output_tensor.ndim == 1):
                        ranks = np.argsort(output_tensor)[::-1]
                        output_tensor = ranks[0:self.topK]
                    output_data[img_name].append(output_tensor)
        return output_data


