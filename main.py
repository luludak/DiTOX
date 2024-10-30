import os
import argparse
import onnx
import onnxruntime
import json
import time
from os import listdir
from os.path import isfile, join #, exists, normpath, basename
from helpers.model_helper import load_config
from benchmarking.model_benchmarking import benchmark
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantFormat, QuantType
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error, compute_weight_error,
    create_activation_matching, create_weight_matching,
    modify_model_output_intermediate_tensors)
from helpers.model_helper import get_size

from readers import input_reader
from runners.tvm_runner import TVMRunner
from runners.onnx_runner import ONNXRunner

def _generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"


def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    calibr_images_folder = script_dir + '/images/one-hundred/'
    images_folder = script_dir + '/images/one-hundred/'
    small_calibr_images_folder = script_dir + '/images/ten'
    

    # Generic config.
    config = load_config('./config.json')

    # Common to all models configuration
    device_name = config["tvm"]["devices"]["selected"]
    build = config["tvm"]["devices"][device_name]

    tvm_runner = TVMRunner({"build": build})
    onnx_runner = ONNXRunner({})

    # Process input parameters and setup model input data reader
    #args = get_args()
    float_model_path = script_dir + '/models/ResNet101_torch.onnx' # args.float_model
    qdq_model_path = script_dir + '/models/out/ResNet101_torch_quant.onnx' # args.qdq_model
    model_config = load_config(float_model_path)
    # (self, onnx_path, output_model_path, input_shape)
    shape = model_config["shape"]
    input_dimension = model_config["input_dimension"]
    # tvm_runner.build_tvm(qdq_model_path, script_dir + "/models/tvm", shape)
    # return

    images_paths = [join(images_folder, f) for f in listdir(images_folder) \
                if isfile(join(images_folder, f))]
    
    calibration_dataset_path = calibr_images_folder #args.calibrate_dataset
    small_calibration_dataset_path = small_calibr_images_folder

    input_data_reader = input_reader.InputReader(
        calibration_dataset_path, float_model_path, model_config["model_name"]
    )

    small_input_data_reader = input_reader.InputReader(
        small_calibration_dataset_path, float_model_path, model_config["model_name"]
    )

    #Calibrate and quantize model
    # Turn off model optimization during quantization
    # quantize_dynamic(
    #     float_model_path,
    #     qdq_model_path,
    #     # input_data_reader,
    #     # quant_format=QuantFormat.QDQ,
    #     # per_channel=False,
    #     weight_type=QuantType.QUInt8
    # )
    # quantize_static(
    #     float_model_path,
    #     qdq_model_path,
    #     input_data_reader,
    #     quant_format=QuantFormat.QDQ,
    #     per_channel=False,
    #     weight_type=QuantType.QInt8
    # )
    # print("Calibrated and quantized model saved.")

    print ("Running original model...")
    base_model_out = onnx_runner.execute_onnx_model(onnx.load(float_model_path), images_paths, config={
        "input_shape": shape,
        "input_dimension": input_dimension,
        "model_name": model_config["model_name"]
    })

    total_dissimilar_percentage = 100
    prev_dissimilar = 100
    threshold = config["onnx"]["threshold"]
    considered_nodes = []
    nodes_to_exclude = [] #["/features/features.0/features.0.0/Conv", "/classifier/classifier.1/Gemm", "classifier.1.weight"]
    new_quant_model_path = qdq_model_path

    float_model = onnx.load(float_model_path)

    quantize_static(
        float_model_path,
        qdq_model_path,
        input_data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude
    )

    # nodes_to_exclude = [n.name for n in float_model.graph.node]

    aug_float_model_path = _generate_aug_model_path(float_model_path)
    modify_model_output_intermediate_tensors(float_model_path, aug_float_model_path)
    small_input_data_reader.rewind()
    float_activations = collect_activations(aug_float_model_path, small_input_data_reader)

    aug_qdq_model_path = _generate_aug_model_path(qdq_model_path)
    modify_model_output_intermediate_tensors(qdq_model_path, aug_qdq_model_path)
    small_input_data_reader.rewind()
    qdq_activations = collect_activations(aug_qdq_model_path, small_input_data_reader)

    act_matching = create_activation_matching(qdq_activations, float_activations)
    act_error = compute_activation_error(act_matching)

    actlist = sorted(act_error.items(), key = lambda x: x[1]['xmodel_err'], reverse=True)

    matched_weights = create_weight_matching(float_model_path, new_quant_model_path)
    weights_error = compute_weight_error(matched_weights)
    
    wlist = sorted(weights_error.items(), key = lambda x: x[1], reverse=True)
    base_benchmark = benchmark(float_model_path, model_config)

    all_dissimilarities = []
    quantized_benchmark = []
    quantized_sizes = []

    result_object = {
        "activations": actlist,
        "weights": wlist,
        "excluded_nodes": nodes_to_exclude,
        "dissimilarities": all_dissimilarities,
        "benchmarks": {
            "original": base_benchmark,
            "quantized": quantized_benchmark
        },
        "size": {
            "original": get_size(float_model_path),
            "quantized": quantized_sizes
        }
    }

    actlist = actlist + wlist

    while True:

        print("Building Quantized Model: " + new_quant_model_path)
        print(nodes_to_exclude)

        input_data_reader.rewind()
        quantize_static(
            float_model_path,
            new_quant_model_path,
            input_data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=nodes_to_exclude
        )

        print ("Running quantized model...")
        quant_model_out = onnx_runner.execute_onnx_model(onnx.load(new_quant_model_path), images_paths, config={
            "input_shape": shape,
            "input_dimension": input_dimension,
            "model_name": model_config["model_name"]
        })

        # print(quant_model_out)

        evaluation = onnx_runner.evaluate(base_model_out, quant_model_out)

        dissimilar_percentage = evaluation["percentage_dissimilar"]

        all_dissimilarities.append(dissimilar_percentage)
        quantized_benchmark.append(benchmark(new_quant_model_path, model_config))
        quantized_sizes.append(get_size(new_quant_model_path))

        print("Dissimilarity: " + str(dissimilar_percentage))

        if (dissimilar_percentage <= total_dissimilar_percentage):
            total_dissimilar_percentage = dissimilar_percentage

            if (dissimilar_percentage <= threshold):
                print("Threshold reached.")
                break
        
        # else:
        #     node_out = nodes_to_exclude.pop()
        #     print (node_out + " removed from list to ignore upon quantization.")
            
        # prev_dissimilar = dissimilar_percentage
        # print("benchmarking fp32 model...")
        # benchmark(float_model_path, model_config)

        # print("------------------------------------------------\n")
        # print("benchmarking int8 model...")
        # benchmark(qdq_model_path, model_config)

        # quantize_dynamic(float_model_path, qdq_model_path, weight_type=QuantType.QUInt8)

        # print("------------------------------------------------\n")
        # print("Comparing weights of float model vs qdq model.....")


        # print(wlist)
        for item in actlist:
            if item[0] not in considered_nodes:
                considered_nodes.append(item[0])
                nodes = onnx_runner.get_nodes_containing_input(float_model, item[0])
                if nodes is []:
                    continue

                for node in nodes:
                    nodes_to_exclude.append(node.name)
                break

        new_quant_model_path = new_quant_model_path.split("quant")[0] + "quant_" + str(time.time()) + ".onnx"
        

    # print("Nodes Excluded:")
    # print(nodes_to_exclude)

    result_object["final_model_path"] = new_quant_model_path

    out_json = json.dumps(result_object, indent=2)
    with open(float_model_path.replace(".onnx", "_out.json"), "w") as outfile:
        outfile.write(out_json)
    
    # print("------------------------------------------------\n")
    # print("Augmenting models to save intermediate activations......")

    # aug_float_model_path = _generate_aug_model_path(float_model_path)
    # modify_model_output_intermediate_tensors(float_model_path, aug_float_model_path)

    # aug_qdq_model_path = _generate_aug_model_path(qdq_model_path)
    # modify_model_output_intermediate_tensors(qdq_model_path, aug_qdq_model_path)

    # print("------------------------------------------------\n")
    # print("Running the augmented floating point model to collect activations......")
    # input_data_reader.rewind()
    # float_activations = collect_activations(aug_float_model_path, input_data_reader)

    # print("------------------------------------------------\n")
    # print("Running the augmented qdq model to collect activations......")
    # input_data_reader.rewind()
    # qdq_activations = collect_activations(aug_qdq_model_path, input_data_reader)

    # print("------------------------------------------------\n")
    # print("Comparing activations of float model vs qdq model......")

    # act_matching = create_activation_matching(qdq_activations, float_activations)
    # act_error = compute_activation_error(act_matching)


    # actlist = sorted(act_error.items(), key = lambda x: x[1]['qdq_err'], reverse=True)
    # print(actlist)

if __name__ == "__main__":
    main()