import os
import json
import hashlib
import subprocess
import numpy as np
from os import listdir
from pathlib import Path
from os.path import isfile, join
from helpers.model_helper import load_config
import onnx
from onnx import hub

from helpers.optimizer_helper import OptimizerHelper
from helpers.model_helper import get_size, get_model_path, load_config, \
    prepare_model_input_dimension, prepare_model_shape

from helpers.yolov2_extractor import YOLOV2Extractor
from helpers.ssd_extractor import SSDExtractor

from evaluators.object_detection import ObjectDetectionEvaluator

from comparators.classification import ClassificationComparator
from comparators.ssd import SSDComparator
from comparators.yolov3 import YOLOV3Comparator

from runners.onnx_runner import ONNXRunner

def main():

    config = load_config('./config.json')
    images_config = config["images"]
    general_config = config["general"]
    hub_config = config["hub"]
    optimizer_config = config["optimizer"]

    include_certainties = general_config["include_certainties"]
    enable_kt_on_one_dim_tensor = general_config["enable_kt_on_one_dim_tensor"]

    script_dir = os.path.dirname(os.path.realpath(__file__))
    images_folder = script_dir + "/" + images_config["images_folder_rel_path"]

    all_models = hub.list_models(tags=hub_config["tags"])

    hub.set_dir(script_dir + "/" + hub_config["cache_folder_rel_path"])
    
    model_comparisons_file = script_dir + "/" + general_config["report_file_base_rel_path"]
    
    onnx_runner = ONNXRunner({})
    opt_helper = OptimizerHelper()

    check_prob_models = general_config["check_reports_for_problematic_models"]
    model_comparisons_file_exists = os.path.exists(model_comparisons_file)
    if check_prob_models:
        if model_comparisons_file_exists:
            # Use model comparisons file to compare problematic models.
            f = open(model_comparisons_file, "r")
            base_object = json.load(f)

            different_models = []
                
            for key in base_object:
                elem = base_object[key]
                if isinstance(elem, dict) and "different" in elem and elem["different"] != 0:
                    different_models.append(key)
            print(different_models)
        else:
            print("Warning: check for problematic models is enabled but comparisons file does not exist. Skipping...")
 
    skip_until = general_config["skip_until_model_id"]
    run_up_to = general_config["run_up_to_model_id"]

    all_images_paths = [join(images_folder, f) for f in listdir(images_folder) \
        if isfile(join(images_folder, f))]

    all_images_paths.sort()

    images_chunk = images_config["images_chunk"]
    starts_from = images_config["starts_from"]
    limit = images_config["limit"]
    ends_at = starts_from + images_chunk

    while ends_at < limit + images_chunk:

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

        # TODO: REFACTOR - ADD CONDITION FOR TEXT NOT IN TAGS.
        images_paths = [x for x in all_images_paths[starts_from:ends_at]]
        print(" ---- Input Batch: " + str(starts_from) + " to " + str(ends_at))
        model_comparisons_file = model_comparisons_file.split("_chunk")[0] + "_chunk_" + str(starts_from) + "_" + str(ends_at) + ".json"

        model_no = 0
        for model_obj in all_models:

            model_no += 1
            if (model_no < skip_until or model_no > run_up_to):
                continue

            # TODO: Add logic to extract model ID from ONNX Hub.
            # if model_obj.model != "GoogleNet-int8":
            #     continue
            # else:
            #     print(model_no)
            #     return

            model_name = model_obj.model
            model_opset = model_obj.opset
            model_name_opset = model_name + "-" + str(model_opset)
            tags = model_obj.tags

            name_filters = general_config["name_filters"]
            
            if len(name_filters) != 0:
                filter_in_name = False
                for filter in name_filters:
                    if filter in model_name_opset:
                        filter_in_name = True
                        break

                if not filter_in_name or (filter_in_name and ("int8" in model_name_opset or "SSD-MobilenetV1" in model_name_opset or "Tiny" in model_name_opset)):
                    continue

            # print("TEST")
            # Check different models if option is enabled and file exists.
            if (check_prob_models and model_comparisons_file_exists and \
                model_name_opset not in different_models):
                continue
                
            tag_found = False
            if len(hub_config["tags_of_models_to_consider"]) != 0:
                for tag_to_consider in hub_config["tags_of_models_to_consider"]:
                    if tag_to_consider in tags:
                        tag_found = True
                        break

                if not tag_found:
                    continue

            if model_opset < hub_config["opset_bottom_threshold"]:
                continue

            print("Model Number: " + str(model_no))
            print("Model Name: " + model_name)
            print(model_obj)

            img_dimension = prepare_model_input_dimension(model_obj)

            keywords = ["tf", "keras", "torch", "tflite", "densenet", \
                "resnet", "mobilenet", "inception", "shufflenet", "googlenet", "version-rfb"]

            # Check if any known library is in name.
            # If not, PyTorch preprocessing configuration will be used.
            contains_keyword = False
            for keyword in keywords:
                if keyword in model_name.lower():
                    contains_keyword = True
                    break

            first_input = model_obj.metadata["io_ports"]["inputs"][0]

            print (first_input["shape"])

            model_shape = prepare_model_shape(model_obj)

            model_config = {
                "model_name": model_name + "_torch" if not contains_keyword else "",
                "input_name": first_input["name"],
                "input_shape": model_shape,
                "input_dimension": img_dimension
            }

            model_path = get_model_path(script_dir, model_obj)

            model = hub.load(model_name, opset=model_opset)

            original_hash = hashlib.md5(open(model_path,'rb').read()).hexdigest()
            opt_model_path = model_path.replace(".onnx", "_opt.onnx")

            # Load passes.
            passes = optimizer_config["passes"] if len(optimizer_config["passes"]) != 0 else \
                opt_helper.get_optimizer_passes()

            print("Running original model...")
            print(model_path)
            base_model_out = None

            try:
                if "text" not in tags:
                    base_model_out = onnx_runner.execute_onnx_model(model, images_paths, config=model_config, include_certainties=include_certainties)
                else:
                    print("Text models not supported on this version.")
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

            run_individual_passes = general_config["run_individual_passes"]

            for current_pass in passes:
                conversion_failed = False
                try:
                    if not run_individual_passes:
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

                except subprocess.CalledProcessError as e:
                    print('Fatal error: code={}, out="{}"'.format(e.returncode, e.output))

                opt_hash = None
                try:
                    opt_hash = hashlib.md5(open(opt_model_path,'rb').read()).hexdigest()
                except:
                    conversion_failed = True
                if (opt_hash is not None and original_hash == opt_hash):
                    print(current_pass + " has no effect on model " + model_name)
                    model_comparisons[model_name_opset]["skipped"].append(current_pass)
                elif (not conversion_failed):
                    print("Running optimized model. Pass: " + current_pass)
                    print(opt_model_path)
                    try:
                        onnx_model = onnx.load(opt_model_path)
                        
                        if "text" not in tags:
                            opt_model_out = onnx_runner.execute_onnx_model(onnx_model, images_paths, config=model_config, include_certainties=include_certainties)

                    except Exception as e:
                        if model_name_opset not in model_comparisons["conversion_errors"]:
                            model_comparisons["conversion_errors"][model_name_opset] = {}
                        model_comparisons["conversion_errors"][model_name_opset][current_pass] = repr(e)
                        model_comparisons[model_name_opset]["failed"].append(current_pass)
                        conversion_failed = True

                    if not conversion_failed:
                        
                        if ("classification" in tags):
                            evaluation = onnx_runner.evaluate(base_model_out, opt_model_out, type=tags, include_certainties=include_certainties)
                            comparator = ClassificationComparator(evaluation, model_comparisons)
                            comparator.update(model_name_opset, current_pass, include_certainties=include_certainties)

                        elif ("object detection segmentation" in tags):
                            model_comparisons[model_name_opset][current_pass] = {}
                            evaluation = onnx_runner.evaluate(base_model_out, opt_model_out, type=tags, include_certainties=include_certainties, enable_kt_on_one_dim_tensor=enable_kt_on_one_dim_tensor)

                            if "YOLOv3" in model_name_opset:
                                comparator = YOLOV3Comparator(evaluation, model_comparisons)
                                comparator.update(model_name_opset, model, base_model_out, opt_model_out, current_pass)

                            elif "YOLOv2-9" in model_name_opset:
                                extractor = YOLOV2Extractor()

                                metrics_5 = []
                                metrics_7 = []
                                metrics_9 = []
                                evaluator_5 = ObjectDetectionEvaluator(iou_threshold=0.5)
                                evaluator_7 = ObjectDetectionEvaluator(iou_threshold=0.75)
                                evaluator_9 = ObjectDetectionEvaluator(iou_threshold=0.9)

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
                                        "avg_mAP": np.mean([o["mAP"] for o in metrics_5]) * 100,
                                        "avg_IoU": np.mean([o["IoU"] for o in metrics_5]) * 100,
                                    },
                                    "metrics_0_7": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_7]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_7]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_7]) * 100,
                                        "avg_mAP": np.mean([o["mAP"] for o in metrics_7]) * 100,
                                        "avg_IoU": np.mean([o["IoU"] for o in metrics_7]) * 100,
                                    },
                                    "metrics_0_9": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_9]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_9]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_9]) * 100,
                                        "avg_mAP": np.mean([o["mAP"] for o in metrics_9]) * 100,
                                        "avg_IoU": np.mean([o["IoU"] for o in metrics_9]) * 100,
                                    }
                                }
                                # comparator = SSDComparator(evaluation, model_comparisons)
                                # comparator.update(model_name_opset, current_pass, include_certainties=include_certainties, array_index=1)

                            elif "SSD" in model_name_opset:
                                extractor = SSDExtractor()

                                metrics_5 = []
                                metrics_7 = []
                                metrics_9 = []
                                evaluator_5 = ObjectDetectionEvaluator(iou_threshold=0.5)
                                evaluator_7 = ObjectDetectionEvaluator(iou_threshold=0.75)
                                evaluator_9 = ObjectDetectionEvaluator(iou_threshold=0.9)

                                output = [node.name for node in model.graph.output]

                                for img_key in base_model_out.keys():
                                    base_image = base_model_out[img_key]
                                    opt_image = opt_model_out[img_key]

                                    (base_bboxes, base_labels, base_scores) = extractor.extract_ssd_outputs(base_image)
                                    (opt_bboxes, opt_labels, opt_scores) = extractor.extract_ssd_outputs(opt_image)

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
                                        "avg_mAP": np.mean([o["mAP"] for o in metrics_5]) * 100,
                                        "avg_IoU": np.mean([o["IoU"] for o in metrics_5]) * 100,
                                    },
                                    "metrics_0_7": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_7]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_7]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_7]) * 100,
                                        "avg_mAP": np.mean([o["mAP"] for o in metrics_7]) * 100,
                                        "avg_IoU": np.mean([o["IoU"] for o in metrics_7]) * 100,
                                    },
                                    "metrics_0_9": {
                                        "avg_f1": np.mean([o["F1"] for o in metrics_9]) * 100,
                                        "avg_prec": np.mean([o["precision"] for o in metrics_9]) * 100,
                                        "avg_recall": np.mean([o["recall"] for o in metrics_9]) * 100,
                                        "avg_mAP": np.mean([o["mAP"] for o in metrics_9]) * 100,
                                        "avg_IoU": np.mean([o["IoU"] for o in metrics_9]) * 100,
                                    }
                                }

                                cmp_object = {}
                                if (evaluation["percentage_dissimilar1"][1] != -1):
                                    cmp_object["first"] = evaluation["percentage_dissimilar1"][1]
                                if (evaluation["percentage_dissimilar5"][1] != -1):
                                    cmp_object["top5"] = evaluation["percentage_dissimilar5"][1]

                                comparator = SSDComparator(evaluation, model_comparisons)
                                comparator.update(model_name_opset, current_pass, include_certainties=include_certainties, array_index=1)
                        else:
                            evaluation = onnx_runner.evaluate(base_model_out, opt_model_out, type=tags, include_certainties=include_certainties)
                            output =[node.name for node in model.graph.output]
                            model_comparisons[model_name_opset][current_pass] = {}

                            for i, output_node in enumerate(output):
                                cmp_object = {}
                                if (evaluation["percentage_dissimilar1"][i] != -1):
                                    cmp_object["first"] = evaluation["percentage_dissimilar1"][i]
                                if (evaluation["percentage_dissimilar5"][i] != -1):
                                    cmp_object["top5"] = evaluation["percentage_dissimilar5"][i]
                                cmp_object["topK"] = evaluation["percentage_dissimilar"][i]

                                model_comparisons[model_name_opset][current_pass][output[i]] = cmp_object

                            # This assumes the new model has the same output structure
                            # with SSD (boxes, classes, scores).
                            # TODO: Extend to work for every model.
                            comparator = SSDComparator(evaluation, model_comparisons)
                            comparator.update(model_name_opset, current_pass, include_certainties=include_certainties, array_index=1)
                            
                        model_comparisons["model_instances_run"] += 1
                        model_comparisons[model_name_opset]["run"] += 1
                    
                    json_object = json.dumps(model_comparisons, indent=2)

                    # TODO: Remove after debugging.
                    with open(model_comparisons_file, "w") as outfile:
                        outfile.write(json_object)
                
                if not run_individual_passes:
                    break

            if json_object is not None:
                with open(model_comparisons_file, "w") as outfile:
                    outfile.write(json_object)
        
        starts_from += images_chunk
        ends_at += images_chunk

if __name__ == "__main__":
    main()