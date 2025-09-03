import os
from .comparator import *
import scipy.stats as stats
import numpy as np
from numpy import array
from decimal import Decimal

class Evaluator:
    def __init__(self, topK=5):
        self.topK = topK
        pass

    def  evaluate_objects(self, source_object, target_object, off_by_one=False, include_certainties=False):

        source_cert = []
        target_cert = []

        if (array(source_object).shape[0] == 1):
            source_object = list(np.squeeze(source_object))
            target_object = list(np.squeeze(target_object))

        source_preds = source_object
        target_preds = target_object

        if include_certainties:

            if len(source_object) > 1:
                source_cert = source_object
                target_cert = target_object
            else:

                source_cert = [t[1] for t in source_object]
                target_cert = [t[1] for t in target_object]
            
                source_preds = [t[0] for t in source_object]
                target_preds = [t[0] for t in target_object]
        

        if(off_by_one):
            target_preds = [int(t) - 1 for t in target_preds]

        first_only = 1 if (source_preds[0] == target_preds[0]) else 0
        
        tau, p_value = stats.kendalltau(source_preds, target_preds)

        tau5, p_value5 = (None, None)
        if len(source_preds) >= 5:
            tau5, p_value5 = stats.kendalltau(source_preds[0:5], target_preds[0:5])
        
        obj_to_return = {
            "base_label1": source_preds[0],
            "eval_label1": target_preds[0],
            "comparisons": {
               "kendalltau": {
                    "tau": tau,
                    "p-value": p_value
                },
                "kendalltau5": {
                    "tau": tau5,
                    "p-value": p_value5
                },
                "first_only": first_only
            },
            
        }

        # Convert float arrays to Decimals safely using str()
        s_dec = [Decimal(str(x)) for x in source_cert]
        t_dec = [Decimal(str(y)) for y in target_cert]

        print(include_certainties)

        # Compute absolute differences and their average
        if include_certainties:

            avg_diff5 = 0
            avg_diff = 0
            avg_diff1 = 0
            if len(s_dec) >= 5:
                abs_diffs5 = [(t - s) / (s if s != Decimal("0") else Decimal("1e-10")) \
                    for s, t in zip(s_dec[0:5], t_dec[0:5]) if (t - s) != Decimal("0")]
                
                avg_diff5 = (sum(abs_diffs5) / Decimal(len(abs_diffs5))) if len(abs_diffs5) != 0 else 0


            # Compute absolute differences and their average
            abs_diffs = [(t - s) / (s if s != Decimal("0") else Decimal("1e-10")) \
                        for s, t in zip(s_dec, t_dec) if (t - s) != Decimal("0")]
            
            if len(abs_diffs) != 0:
                avg_diff = (sum(abs_diffs) / Decimal(len(abs_diffs))) if len(abs_diffs) != 0 else 0


            avg_diff1 = (t_dec[0] - s_dec[0]) / (s_dec[0] if s_dec[0] != Decimal("0") else Decimal("1e-10"))

            # Divide by epsilon if source value is zero.
            obj_to_return["certainties"] = {
                "top1": avg_diff1,
                "top5": avg_diff5,
                "topK": avg_diff
            }

        return obj_to_return


    def evaluate(self, original_file_path, mutant_file_path):

        original_obj, lines_no, original_first_line, exec_time1 = self.file_to_object(original_file_path)
        mutant_obj, mutant_lines_no, mutant_first_line, exec_time2 = self.file_to_object(mutant_file_path)
        
        original_keys = map(lambda x: int(x), original_obj.keys())
        mutant_keys = map(lambda x: int(x), mutant_obj.keys())

        total_value = 0

        first_only = 1 if (original_first_line == mutant_first_line) else 0

        for mutant_class in mutant_obj:
            class_value = 0

            if mutant_class in original_obj:
                
                class_value = 1 - float(abs(original_obj[mutant_class]["order"] - mutant_obj[mutant_class]["order"])) / (lines_no*2)
            total_value += class_value

        total_value_percentage = (float(total_value)/lines_no)*100
        original_list = list(original_obj.keys())
        mutants_list = list(mutant_obj.keys())

        tau, p_value = stats.kendalltau(original_list, mutants_list)

        tau5, p_value5 = (None, None)
        if len(original_list) >= 5:
            tau5, p_value5 = stats.kendalltau(original_list[0:5], mutants_list[0:5])
        
        return {
            "path_to_file": mutant_file_path,
            "base_comparison_file": original_file_path,
            "base_label1": original_first_line,
            "eval_label1": mutant_first_line,
            "base_exec_time": exec_time1,
            "exec_time": exec_time2,
            "comparisons": {
                "jaccard": str(jaccard_similarity(original_obj.keys(), mutant_obj.keys())),
                "euclideanDistance" :  str(euclidean_distance(original_keys, mutant_keys)),
                "manhattanDistance": str(manhattan_distance(original_keys, mutant_keys)),
                "minkowskiDistance": str(minkowski_distance(original_keys, mutant_keys, 1.5)), # p values: 1 is for Manhattan, 2 is for Euclidean. Set it in between.
                "kendalltau": {
                    "tau": str(tau),
                    "p-value": str(p_value)
                },
                "kendalltau5": {
                    "tau": str(tau5),
                    "p-value": str(p_value5)
                },
                "custom": str(total_value_percentage),
                "first_only": str(first_only),
                "rbo": str(rbo(list(original_obj.keys()), list(mutant_obj.keys()), 0.8))
            }
            
        }
        
    def compare_to_original(self, original_file_path, mutant_file_path):

        if (not os.path.exists(original_file_path) or not os.path.exists(mutant_file_path)):
            return False

        original_obj, lines_no, original_first_line, exec_time1 = self.file_to_object(original_file_path)
        mutant_obj, mutant_lines_no, mutant_first_line, exec_time2 = self.file_to_object(mutant_file_path)
        
        
        original_keys = map(lambda x: int(x), original_obj.keys())
        mutant_keys = map(lambda x: int(x), mutant_obj.keys())

        for mutant_class in mutant_obj:
            if mutant_class not in original_obj or (original_obj[mutant_class]["order"] != mutant_obj[mutant_class]["order"]):
                return False
        return True

    def reset_output_file(self, output_file_path):
        output_file = open(output_file_path, 'w')
        output_file.close()

    def file_to_object(self, path):
        obj = {}

        fileObj = open(path, 'r')

        first_class = None

        lines = fileObj.readlines()
        order = 1

        # Set this for Top-K (default: 5).
        count = 0
        exec_time = 0.0
        for line in lines:
            if count < self.topK:
                line_split = line.split(", ")
                class_name = line_split[0]
                
                # Comparison of libraries (does not apply in conversions).
                if(not "_to_" in path):

                    # Torch and Keras-related models correspond to N classes, while TF/TFLite to N+1,
                    # Therefore we subtract 1 from them.
                    if (not "torch" in path and not "keras" in path):
                	    class_name = str(int(class_name) - 1)

                if (count == 0):
                    first_class = class_name
                    
                class_prob = line_split[1]
                obj[class_name] = {
                    "probability": float(class_prob),
                    "order": int(order)
                }
                order += 1
            elif(count == self.topK + 1):
                exec_time = float(line)
            count += 1


        return obj, order - 1, first_class, exec_time
