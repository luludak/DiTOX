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

    def __init__(self, config):
        self.evaluation_generator = EvaluationGenerator()

    def evaluate(self, source_run_obj, target_run_obj):
        return self.evaluation_generator.generate_objects_comparison(source_run_obj, target_run_obj)

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
 
    def execute_single(self, onnx_path, image_path, config, include_certainties=False):
        # Execute and return for single image.
        return self.execute_onnx_path(onnx_path, [image_path], config, include_certainties)

    def execute_onnx_path(self, onnx_path, images_paths, config, image_names=None, print_percentage=True, include_certainties=False): 
        onnx_model = onnx.load(onnx_path)
        return self.execute_onnx_model(onnx_model, images_paths, config, image_names, print_percentage, include_certainties)

    def get_nodes_containing_input(self, base_model, input_name):

        base_nodes = base_model.graph.node
        nodes_found = []
        for base_node in base_nodes:
            if input_name in base_node.input:
                nodes_found.append(base_node)
        return nodes_found


    def execute_onnx_model(self, onnx_model, images_paths, config, image_names=None, print_percentage=True, include_certainties=False):
        # Execute and return all data.
        self.preprocessing_data = {
            "input": config["input_shape"],
            "input_dimension": config["input_dimension"],
            "library": None
        }
        # Set library to None, so that the name is utilized for preprocessing selection.
        model_preprocessor = ModelPreprocessor(self.preprocessing_data)

        ort_sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])

        output_data = {}
        count = 0
        images_length = len(images_paths)
        step = (images_length // 4) if images_length > 4 else images_length

        for img_path in images_paths:
            
            if(print_percentage and count % step == 0):
                print("Complete: " + str((count // step) * 25) + "%")

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
                

            img = img.resize(config["input_dimension"])
            img = model_preprocessor.preprocess(config["model_name"], img, True)

            input_name = onnx_model.graph.input[0].name if "input_name" not in config else config["input_name"]
            # print(img.astype(np.float32))
            output = ort_sess.run(None, {input_name : img.astype(np.float32)})

            if len(output) > 1:
                output = output[2]

            scores = softmax(output)
            if(len(scores) > 2):
                squeeze(scores, [0, 2])
            # elif(len(scores) == 2 and (scores[0] == scores[1]).all()):
            #     scores = scores[0]

            scores = np.squeeze(scores)
            ranks = np.argsort(scores)[::-1]
            # In case of a double output.
            # if(len(ranks) == 2):
            #     ranks = ranks[0]
            extracted_ranks = ranks[0:5]
            # print(extracted_ranks)
            # We do not consider probabilities for now
            if include_certainties:
                output_data[img_name] = [(rank, str(scores[rank])) for rank in extracted_ranks.tolist()]
            else:
                output_data[img_name] = extracted_ranks.tolist()
        return output_data

