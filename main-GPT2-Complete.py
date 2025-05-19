import os
import argparse
import onnx
import onnxruntime
import json
import time
import hashlib
import subprocess
import numpy as np
import onnxruntime as ort
import scipy.stats as stats
import math
import torch

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

from helpers.yolov2_extractor import YOLOV2Extractor
from helpers.yolov3_extractor import YOLOV3Extractor

from evaluators.object_detection_ssd import SSDObjectDetectionEvaluator
from evaluators.object_detection_yolov3 import YOLOV3ObjectDetectionEvaluator

from runners.onnx_runner import ONNXRunner
from onnx.numpy_helper import to_array
from onnx import hub

from datasets import load_dataset

from transformers import GPT2Tokenizer

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def get_model_path(script_dir, model_obj):
    return script_dir + "/models_cache/" + model_obj.model_path.replace("/model/", "/model/" + \
            model_obj.metadata["model_sha"] + "_").replace("/models/", "/models/" + \
            model_obj.metadata["model_sha"] + "_").replace("/preproc/", "/preproc/" + \
            model_obj.metadata["model_sha"] + "_")

def generate_text(session, input_text, tokenizer, max_length=128, temperature=1, top_k=1):
    generated_tokens = []  # Store all logits locally
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="np", padding=False)
    input_ids = inputs["input_ids"].astype(np.int64)
    input_ids = np.expand_dims(input_ids, axis=0)  # Ensure correct shape
    
    for _ in range(max_length):
        outputs = session.run(["output1"], {"input1": input_ids})
        logits = torch.tensor(outputs[0])  # Convert to PyTorch tensor for easier manipulation
        
        # Extract last token in sequence
        logits = logits.squeeze(0)[:, -1, :]
        
        # Apply temperature scaling
        logits /= temperature

        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Top-3 selection always.
        # next_token = torch.topk(probs, k=3, dim=-1).indices[..., 2]
        # Top-2 selection always.
        # next_token = torch.topk(probs, k=2, dim=-1).indices[..., 1]
        # Top-1 selection always.
        torch.argmax(probs, dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append the generated token
        generated_tokens.append(next_token.item())
        next_token_reshaped = np.array(next_token.numpy().reshape(1, 1, 1))  # Ensure correct shape
        input_ids = np.concatenate([input_ids, next_token_reshaped], axis=-1)  # Keep window size fixed

    # Decode full generated sequence
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Decode final sequence
    return decoded_text

def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    calibr_images_folder = script_dir + '/images/one-hundred/'
    images_folder = script_dir + '/images/imagenet'
    small_calibr_images_folder = script_dir + '/images/one'

    all_models = hub.list_models(tags=["text"])

    hub.set_dir(script_dir + "/models_cache")
    
    model_comparisons_file = script_dir + "/text/Top2Prediction_GPT-2_chunk_0_500.json"
    onnx_runner = ONNXRunner({})
    skip_until = 8
    run_up_to = 8

    all_images_paths = [join(images_folder, f) for f in listdir(images_folder) \
        if isfile(join(images_folder, f))]

    all_images_paths.sort()

    images_chunk = 200
    starts_from = 10600
    limit = 11873 # 25000
    ends_at = starts_from + images_chunk
    tags = "text"

    dataset = load_dataset("squad_v2")

    print(len(dataset["validation"]))

    while ends_at < limit + images_chunk:
        # BERT-Squad
        qa_data = list(dataset["validation"])[starts_from:ends_at]
        total_qa = ['Context: ' + qa["context"] + ' Question: ' + qa["question"] for qa in qa_data]

        json_object = None
        model_comparisons = {}
        model_comparisons["no_dissimilar"] = 0
        model_comparisons["base_models_run"] = 0
        model_comparisons["model_instances_run"] = 0
        model_comparisons["failed_conversions_no"] = 0
        model_comparisons["skipped_models"] = []
        model_comparisons["failed_models"] = []
        model_comparisons["conversion_errors"] = {}

        images_paths = None
        inputs = None

        if "text" not in tags:
            images_paths = [x for x in all_images_paths[starts_from:ends_at]]

        print(" ---- Input Batch: " + str(starts_from) + " to " + str(ends_at))
        model_comparisons_file = model_comparisons_file.split("_chunk")[0] + "_chunk_" + str(starts_from) + "_" + str(ends_at) + ".json"

        model_no = 0
        for model_obj in all_models:
            text_inputs = None
            text_expected_outputs = None

            
            model_no += 1
            if (model_no < skip_until or model_no > run_up_to):
                continue

            model_name = model_obj.model
            model_opset = model_obj.opset
            model_name_opset = model_name + "-" + str(model_opset)
            tags = model_obj.tags


            if "text" in tags:
                model_path_data = hub.download_model_with_test_data(model_name)

                data_path = model_path_data + "/test_data_set_0/"
                data_path = Path(data_path)
                model_path = list(Path(model_path_data + "/GPT-2-LM-HEAD").glob("*.onnx"))

                print ("PATH: " + model_path_data + "/GPT-2-LM-HEAD")

                if len(model_path) == 0:
                    continue

                model_path = [f for f in model_path if "_opt" not in str(f)]
                model_path = str(model_path[0])
                
                # Load GPT-2 tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token

            print("Model Number: " + str(model_no))
            print("Model Name: " + model_name)
            print(model_obj)

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
            # if "text" not in tags:
            if os.path.exists(model_path):
                print(model_path)
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
            base_match_percentage = 0

            base_answers_out = []
            base_outputs = None

            session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            try:
                if ("gpt-2" in tags):
                    for tq in total_qa:
                        out = generate_text(session, tq, tokenizer)
                        base_answers_out.append(out)
                else:
                    return
            except Exception as e:
                print(e)
                print(model_name + " - a base model error occured!")
                model_comparisons["failed_models"].append(model_name)
                continue

            model_comparisons["base_models_run"] += 1

            model_comparisons[model_name_opset] = {
                "skipped": [],
                "failed": [],
                "different": 0,
                "run": 0
            }

            basic_run = True

            for current_pass in passes:
                opt_match_percentage = 0
                matches = 0
                total = 0
                tau_values = []
                p_values = []
                opt_answers_out = []
                all_bleu_scores = []
                conversion_failed = False
                try:
                    if basic_run:
                        p = subprocess.Popen(['python3 -m onnxoptimizer ' + model_path + " " + opt_model_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                        current_pass = "all"
                    else:
                        p = subprocess.Popen(['python3 -m onnxoptimizer ' + model_path + " " + opt_model_path + " -p " + current_pass], stdout=subprocess.PIPE, \
                                             stderr=subprocess.STDOUT, shell=True)
                    
                    (output, err) = p.communicate()  
                    p.wait()
                    if p.returncode != 0:
                        print(output)
                        if model_name_opset not in model_comparisons["conversion_errors"]:
                            model_comparisons["conversion_errors"][model_name_opset] = {}
                        model_comparisons["conversion_errors"][model_name_opset][current_pass] = repr(output)
                        model_comparisons["failed_conversions_no"] += 1
                        conversion_failed = True

                except subprocess.CalledProcessError as e:
                    print('Fatal error: code={}, out="{}"'.format(e.returncode, e.output))

                opt_hash = None
                try:
                    opt_hash = hashlib.md5(open(opt_model_path,'rb').read()).hexdigest()
                except:
                    conversion_failed = True
                # break
                if (opt_hash is not None and original_hash == opt_hash):
                    print(current_pass + " has no effect on model " + model_name)
                    model_comparisons[model_name_opset]["skipped"].append(current_pass)
                elif (not conversion_failed):
                    # TODO: Add optimizer.
                    print("Running optimized model. Pass: " + current_pass)
                    print(opt_model_path)
                    try:
                        onnx_model = onnx.load(opt_model_path)
                        
                        if "text" not in tags:
                            opt_model_out = onnx_runner.execute_onnx_model(onnx_model, images_paths, config=model_config)
                        else:
                            # TODO: move to runner.
                            session = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
                            for tq in total_qa:
                                # print(tq)
                                out = generate_text(session, tq, tokenizer)
                                opt_answers_out.append(out)


                            
                            for i, b_out in enumerate(base_answers_out):
                                opt_out = opt_answers_out[i]
                                # Tokenize the sentences
                                prediction_tokens = b_out.split()
                                reference_tokens = opt_out.split()
                                
                                # Calculate BLEU score
                                bleu_score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=SmoothingFunction().method4)
                                all_bleu_scores.append(bleu_score)

                    except Exception as e:
                        print(e)
                        if model_name_opset not in model_comparisons["conversion_errors"]:
                            model_comparisons["conversion_errors"][model_name_opset] = {}
                        model_comparisons["conversion_errors"][model_name_opset][current_pass] = repr(e)
                        model_comparisons[model_name_opset]["failed"].append(current_pass)
                        conversion_failed = True

                    if not conversion_failed:
                        
                        if ("gpt-2" in tags):
                            print(all_bleu_scores)
                            model_comparisons[model_name_opset][current_pass] = {
                                "mean_bleu": np.mean(all_bleu_scores),
                                "median_bleu": np.median(all_bleu_scores)
                            }

                        elif ("text" in tags):
                            model_comparisons[model_name_opset][current_pass] = {
                                "percentage_similarity_diff": opt_match_percentage,
                                "matches": matches,
                                "total": total
                            }

                            if len(tau_values) != 0:
                                model_comparisons[model_name_opset][current_pass]["avg_tau"] = np.mean(tau_values)
                                model_comparisons[model_name_opset][current_pass]["median_tau"] = np.median(tau_values)

                            if len(p_values) != 0:
                                model_comparisons[model_name_opset][current_pass]["avg_pv"] = np.mean(tau_values)
                                model_comparisons[model_name_opset][current_pass]["median_pv"] = np.median(tau_values)


                        elif ("classification" in tags):
                            evaluation = onnx_runner.evaluate(base_model_out, opt_model_out, type=tags)
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
                        elif ("object detection segmentation" in tags):
                            # Use ONLY for labels.
                            model_comparisons[model_name_opset][current_pass] = {}
                            evaluation = onnx_runner.evaluate(base_model_out, opt_model_out, type=tags)
                            if "YOLOv3" in model_name_opset:

                                extractor = YOLOV3Extractor()

                                metrics_5 = []
                                metrics_7 = []
                                metrics_9 = []
                                evaluator_5 = YOLOV3ObjectDetectionEvaluator(iou_threshold=0.5)
                                evaluator_7 = YOLOV3ObjectDetectionEvaluator(iou_threshold=0.75)
                                evaluator_9 = YOLOV3ObjectDetectionEvaluator(iou_threshold=0.9)

                                output = [node.name for node in model.graph.output]

                                for img_key in base_model_out.keys():
                                    base_image = base_model_out[img_key]
                                    opt_image = opt_model_out[img_key]
                                    base_data = extractor.extract_yolo_outputs(base_image)
                                    base_bboxes = base_data[0]
                                    base_labels = base_data[1]
                                    base_scores = base_data[2]

                                    opt_data = extractor.extract_yolo_outputs(opt_image)
                                    opt_bboxes = opt_data[0]
                                    opt_labels = opt_data[1]
                                    opt_scores = opt_data[2]

                                    # Skip undetected classes.
                                    if (base_labels == [] and opt_labels == []) or (base_bboxes == [] and opt_bboxes == []):
                                        continue

                                    metric_5 = evaluator_5.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_5.append(metric_5)

                                    metric_7 = evaluator_7.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_7.append(metric_7)

                                    metric_9 = evaluator_9.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_9.append(metric_9)

                                
                                model_comparisons[model_name_opset][current_pass][output[0]] = {
                                    "metrics_0_5": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_5]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_5]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_5]) * 100,
                                    },
                                    "metrics_0_7": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_7]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_7]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_7]) * 100,
                                    },
                                    "metrics_0_9": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_9]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_9]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_9]) * 100,
                                    }
                                }


                                cmp_object = {}
                                if (evaluation["percentage_dissimilar1"][1] != -1):
                                    cmp_object["first"] = evaluation["percentage_dissimilar1"][1]
                                if (evaluation["percentage_dissimilar5"][1] != -1):
                                    cmp_object["top5"] = evaluation["percentage_dissimilar5"][1]
                                cmp_object["topK"] = evaluation["percentage_dissimilar"][1]
                                # Labels
                                model_comparisons[model_name_opset][current_pass][output[1]] = cmp_object
                                
                            elif "YOLOv2-9" in model_name_opset:
                                extractor = YOLOV2Extractor()

                                metrics_5 = []
                                metrics_7 = []
                                metrics_9 = []
                                evaluator_5 = SSDObjectDetectionEvaluator(iou_threshold=0.5)
                                evaluator_7 = SSDObjectDetectionEvaluator(iou_threshold=0.75)
                                evaluator_9 = SSDObjectDetectionEvaluator(iou_threshold=0.9)

                                output = [node.name for node in model.graph.output]

                                for img_key in base_model_out.keys():
                                    base_image = base_model_out[img_key]
                                    opt_image = opt_model_out[img_key]
                                    base_data = extractor.extract_yolo_outputs(base_image)
                                    base_bboxes = base_data[0]
                                    base_labels = base_data[1]
                                    base_scores = base_data[2]

                                    opt_data = extractor.extract_yolo_outputs(opt_image)
                                    opt_bboxes = opt_data[0]
                                    opt_labels = opt_data[1]
                                    opt_scores = opt_data[2]

                                    # Skip undetected classes.
                                    if (base_labels == [] and opt_labels == []) or (base_bboxes == [] and opt_bboxes == []):
                                        continue
                                    # print(opt_bboxes)

                                    metric_5 = evaluator_5.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_5.append(metric_5)

                                    metric_7 = evaluator_7.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_7.append(metric_7)

                                    metric_9 = evaluator_9.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_9.append(metric_9)

                                model_comparisons[model_name_opset][current_pass][output[0]] = {
                                    "metrics_0_5": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_5]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_5]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_5]) * 100,
                                    },
                                    "metrics_0_7": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_7]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_7]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_7]) * 100,
                                    },
                                    "metrics_0_9": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_9]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_9]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_9]) * 100,
                                    }
                                }


                            elif "SSD-12" in model_name_opset:

                                metrics_5 = []
                                metrics_7 = []
                                metrics_9 = []
                                evaluator_5 = SSDObjectDetectionEvaluator(iou_threshold=0.5)
                                evaluator_7 = SSDObjectDetectionEvaluator(iou_threshold=0.75)
                                evaluator_9 = SSDObjectDetectionEvaluator(iou_threshold=0.9)

                                output = [node.name for node in model.graph.output]

                                for img_key in base_model_out.keys():
                                    base_image = base_model_out[img_key]
                                    opt_image = opt_model_out[img_key]

                                    base_bboxes = np.squeeze(base_image[0]).tolist()
                                    base_labels = np.squeeze(base_image[1]).tolist()
                                    base_scores = np.squeeze(base_image[2]).tolist()

                                    opt_bboxes = np.squeeze(opt_image[0]).tolist()
                                    opt_labels = np.squeeze(opt_image[1]).tolist()
                                    opt_scores = np.squeeze(opt_image[2]).tolist()

                                    if (base_labels == [] and opt_labels == []) or (base_bboxes == [] and opt_bboxes == []):
                                        continue

                                    metric_5 = evaluator_5.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_5.append(metric_5)

                                    metric_7 = evaluator_7.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_7.append(metric_7)

                                    metric_9 = evaluator_9.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
                                    metrics_9.append(metric_9)

                                
                                model_comparisons[model_name_opset][current_pass][output[0]] = {
                                    "metrics_0_5": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_5]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_5]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_5]) * 100,
                                    },
                                    "metrics_0_7": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_7]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_7]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_7]) * 100,
                                    },
                                    "metrics_0_9": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_9]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_9]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_9]) * 100,
                                    }
                                }

                                cmp_object = {}
                                if (evaluation["percentage_dissimilar1"][1] != -1):
                                    cmp_object["first"] = evaluation["percentage_dissimilar1"][1]
                                if (evaluation["percentage_dissimilar5"][1] != -1):
                                    cmp_object["top5"] = evaluation["percentage_dissimilar5"][1]
                                cmp_object["topK"] = evaluation["percentage_dissimilar"][1]
                                # Labels
                                model_comparisons[model_name_opset][current_pass][output[1]] = cmp_object

                        else:
                            evaluation = onnx_runner.evaluate(base_model_out, opt_model_out, type=tags)
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
                    # with open(model_comparisons_file, "w") as outfile:
                    #     outfile.write(json_object)
                
                if basic_run:
                    break

            if json_object is not None:
                with open(model_comparisons_file, "w") as outfile:
                    outfile.write(json_object)
        
        starts_from += images_chunk
        ends_at += images_chunk


if __name__ == "__main__":
    main()
