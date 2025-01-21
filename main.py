import os
import argparse
import onnx
import onnxruntime
import json
import time
import hashlib
import subprocess

os.environ['CURL_CA_BUNDLE'] = ''

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from os import listdir
from pathlib import Path
from os.path import isfile, join
from helpers.model_helper import load_config
from benchmarking.model_benchmarking import benchmark
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantFormat, QuantType
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error, compute_weight_error,
    create_activation_matching, create_weight_matching,
    modify_model_output_intermediate_tensors)
from helpers.model_helper import get_size

# from builders.tvm_builder import TVMBuilder
# from runners.tvm_runner import TVMRunner
from runners.onnx_runner import ONNXRunner

from onnx import hub

# import urllib3
# resp = urllib3.request("GET", "https://github.com/onnx/models/")
# print(resp.status)

def get_model_path(script_dir, model_obj):
    return script_dir + "/models_cache/" + model_obj.model_path.replace("/model/", "/model/" + \
            model_obj.metadata["model_sha"] + "_").replace("/models/", "/models/" + \
            model_obj.metadata["model_sha"] + "_").replace("/preproc/", "/preproc/" + \
            model_obj.metadata["model_sha"] + "_")
        

def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    calibr_images_folder = script_dir + '/images/one-hundred/'
    images_folder = script_dir + '/images/one/'
    small_calibr_images_folder = script_dir + '/images/ten'

    images_paths = [join(images_folder, f) for f in listdir(images_folder) \
            if isfile(join(images_folder, f))]

    all_models = hub.list_models(tags=["vision"])

    hub.set_dir(script_dir + "/models_cache")

    model_comparisons = {}
    model_comparisons["no_dissimilar"] = 0
    model_comparisons["models_run"] = 0
    model_comparisons["model_instances_run"] = 0
    model_comparisons["failed_conversions_no"] = 0
    model_comparisons["skipped_models"] = []
    model_comparisons["failed_models"] = []
    model_comparisons["conversion_errors"] = {}
    model_comparisons_file = script_dir + "/model_comparisons.json"
    base_file = script_dir + "/classification.json"

    onnx_runner = ONNXRunner({})

    f = open(base_file, "r")
    base_object = json.load(f)

    different_models = []
    for key in base_object:
        elem = base_object[key]
        if isinstance(elem, dict) and "different" in elem and elem["different"] != 0:
            different_models.append(key)
    print(different_models)
 
    model_no = 0
    skip_until = 0
    run_up_to = 200

    json_object = None

    for model_obj in all_models:

        # print(str(model_no + 1) + ". " + model_obj.model)
        model_no += 1
        if (model_no < skip_until or model_no > run_up_to):
            continue

        print("Model Number: " + str(model_no))
        model_name = model_obj.model
        model_opset = model_obj.opset
        model_name_opset = model_name + "-" + str(model_opset)
        tags = model_obj.tags

        print("Model Name: " + model_name)
        print(model_obj)
        # TODO: Enable on check.
        # if (model_name_opset not in different_models):
        #     continue

        # if not model_name.startswith("Emotion") or model_opset < 3:
        # TODO: Add rest of models.
        if "object detection segmentation" not in tags or model_opset < 7 or "preproc" in model_name:
            model_comparisons["skipped_models"].append(model_name_opset)
            continue

        model_path = get_model_path(script_dir, model_obj)

        model_shape = [1, 1, 64, 64]
        input_name = None
        if "io_ports" in model_obj.metadata:
            first_input = model_obj.metadata["io_ports"]["inputs"][0]
            model_shape = first_input["shape"]
            input_name = first_input["name"]
        
        if "R-CNN" in model_name:
            model_shape = [1, 3, 224, 224]
        elif "FCN" in model_name or "YOLOv3" in model_name:
            model_shape = [1, 3, 224, 224]

        m_shape = model_shape.copy()
        if (len(m_shape) > 0):
            m_shape.pop(0)

        img_dimension = []
        for img_dim in m_shape:
            if type(img_dim) == str:
                continue
            
            if (img_dim > 5):
                img_dimension.append(img_dim)

        keywords = ["tf", "keras", "torch", "tflite", "densenet", \
            "resnet", "mobilenet", "inception", "shufflenet", "googlenet", "version-rfb"]

        # Check if any known library is in name.
        # If not, PyTorch preprocessing configuration will be used.
        contains_keyword = False
        for keyword in keywords:
            if keyword in model_name.lower():
                contains_keyword = True
                break

        model_config = {
            "model_name": model_name + "_torch" if not contains_keyword else "",
            "input_name": input_name,
            "input_shape": model_shape,
            "input_dimension": img_dimension
        }

        print(model_config)


        if os.path.exists(model_path):
            model = onnx.load(model_path)
        else:
            model = hub.load(model_name, opset=model_opset)

        original_hash = hashlib.md5(open(model_path,'rb').read()).hexdigest()
        opt_model_path = model_path.replace(".onnx", "_opt.onnx")

        passes = ["adjust_add", "rename_input_output", "set_unique_name_for_nodes", \
            "nop", "eliminate_nop_cast", "eliminate_nop_dropout", "eliminate_nop_flatten", \
            "extract_constant_to_initializer", "eliminate_consecutive_idempotent_ops", \
            "eliminate_if_with_const_cond", "eliminate_nop_monotone_argmax", "eliminate_nop_pad", \
            "eliminate_nop_concat", "eliminate_nop_split", "eliminate_nop_expand", "eliminate_shape_gather", \
            "eliminate_slice_after_shape", "eliminate_nop_transpose", "fuse_add_bias_into_conv", "fuse_bn_into_conv", \
            "fuse_consecutive_concats", "fuse_consecutive_log_softmax", "fuse_consecutive_reduce_unsqueeze", "fuse_consecutive_squeezes", \
            "fuse_consecutive_transposes", "fuse_matmul_add_bias_into_gemm", "fuse_pad_into_conv", "fuse_pad_into_pool", "fuse_transpose_into_gemm", \
            "replace_einsum_with_matmul", "lift_lexical_references", "split_init", "split_predict", "fuse_concat_into_reshape", \
            "eliminate_nop_reshape", "eliminate_nop_with_unit", "eliminate_common_subexpression", "fuse_qkv", "fuse_consecutive_unsqueezes", \
            "eliminate_deadend", "eliminate_identity", "eliminate_shape_op", "fuse_consecutive_slices", "eliminate_unused_initializer", \
            "eliminate_duplicate_initializer", "adjust_slice_and_matmul", "rewrite_input_dtype"]

        # TODO: Cache!
        print("Running original model...")
        print(model_path)
        base_model_out = None
        try:
            # pass
            base_model_out = onnx_runner.execute_onnx_model(model, images_paths, config=model_config)
        except Exception as e:
            print(e)
            print(model_name + " - a base model error occured!")
            model_comparisons["failed_models"].append(model_name)
            continue

        model_comparisons["models_run"] += 1

        model_comparisons[model_name_opset] = {
            "skipped": [],
            "failed": [],
            "different": 0,
            "run": 0
        }

        basic_run = True

        for current_pass in passes:
            conversion_failed = False
            try:
                if basic_run:
                    p = subprocess.Popen(['python3 -m onnxoptimizer ' + model_path + " " + opt_model_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                    current_pass = "all"
                else:
                    p = subprocess.Popen(['python3 -m onnxoptimizer ' + model_path + " " + opt_model_path + " -p " + current_pass], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                
                (output, err) = p.communicate()  
                p.wait()
                if p.returncode != 0:
                    print(output)
                    if model_name_opset not in model_comparisons["conversion_errors"]:
                        model_comparisons["conversion_errors"][model_name_opset] = {}
                    model_comparisons["conversion_errors"][model_name_opset][current_pass] = repr(output)
                    model_comparisons["failed_conversions_no"] += 1
                    conversion_failed = True
                    if basic_run:
                        break
                    continue
            except subprocess.CalledProcessError as e:
                print('Fatal error: code={}, out="{}"'.format(e.returncode, e.output))

            try:
                opt_hash = hashlib.md5(open(opt_model_path,'rb').read()).hexdigest()
            except:
                continue
            # break
            if (original_hash == opt_hash):
                print(current_pass + " has no effect on model " + model_name)
                model_comparisons[model_name_opset]["skipped"].append(current_pass)
            elif (not conversion_failed):
                # TODO: Add optimizer.
                print("Running optimized model. Pass: " + current_pass)
                # print(opt_model_path)
                try:
                    onnx_model = onnx.load(opt_model_path)
                    opt_model_out = onnx_runner.execute_onnx_model(onnx_model, images_paths, config=model_config)
                except:
                    print(model_name + " - an optimized model error occured.")
                    model_comparisons[model_name_opset]["failed"].append(current_pass)
                    if basic_run:
                        break
                    continue

                evaluation = onnx_runner.evaluate(base_model_out, opt_model_out, type=tags)

                if ("object detection segmentation" not in tags):
                    dissimilar_percentage1 = evaluation["percentage_dissimilar1"]
                    dissimilar_percentage5 = evaluation["percentage_dissimilar5"]
                    dissimilar_percentage = evaluation["percentage_dissimilar"]
                    print("Dissimilarity for " + current_pass + " (top-1): " + str(dissimilar_percentage1))
                    print("Dissimilarity for " + current_pass + " (top-5): " + str(dissimilar_percentage5))
                    print("Dissimilarity for " + current_pass + " (top-K): " + str(dissimilar_percentage))

                    model_comparisons[model_name_opset][current_pass] = {
                        "first": str(dissimilar_percentage1),
                        "top5": str(dissimilar_percentage5),
                        "topK": str(dissimilar_percentage)
                    }

                    if (dissimilar_percentage > 0):
                        model_comparisons[model_name_opset]["different"] += 1

                    model_comparisons["no_dissimilar"] += 1 if dissimilar_percentage != 0 else 0
                else:

                    output =[node.name for node in model.graph.output]
                    model_comparisons[model_name_opset][current_pass] = {}

                    for i, output_node in enumerate(output):
                    # for i, diss in enumerate(evaluation["percentage_dissimilar"]):
                        cmp_object = {}
                        if (evaluation["percentage_dissimilar1"][i] != -1):
                            cmp_object["first"] = evaluation["percentage_dissimilar1"][i]
                        if (evaluation["percentage_dissimilar5"][i] != -1):
                            cmp_object["top5"] = evaluation["percentage_dissimilar5"][i]
                        cmp_object["topK"] = evaluation["percentage_dissimilar"][i]

                        model_comparisons[model_name_opset][current_pass][output[i]] = cmp_object


                model_comparisons["model_instances_run"] += 1
                model_comparisons[model_name_opset]["run"] += 1
                json_object = json.dumps(model_comparisons, indent=2)

            # TODO: Remove after debugging.
            with open(model_comparisons_file, "w") as outfile:
                outfile.write(json_object)
            
            if basic_run:
                break

        if json_object is not None:
            with open(model_comparisons_file, "w") as outfile:
                outfile.write(json_object)

if __name__ == "__main__":
    main()