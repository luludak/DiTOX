from decimal import Decimal

class SSDComparator:

    def __init__(self, evaluation, comparisons):
        self.evaluation = evaluation
        self.comparisons = comparisons

    def update(self, model_name, current_pass="all", include_certainties=False, array_index=0):
        dissimilar_percentage1 = self.evaluation["percentage_dissimilar1"][array_index]
        dissimilar_percentage5 = self.evaluation["percentage_dissimilar5"][array_index]
        dissimilar_percentage = self.evaluation["percentage_dissimilar"][array_index]
        certainty_diff1 = self.evaluation["certainties"]["top1"]
        certainty_diff5 = self.evaluation["certainties"]["top5"]
        certainty_diff = self.evaluation["certainties"]["topK"]
        
        print("Dissimilarity for " + current_pass + " (top-1): " + str(dissimilar_percentage1))
        print("Dissimilarity for " + current_pass + " (top-5): " + str(dissimilar_percentage5))
        print("Dissimilarity for " + current_pass + " (top-K): " + str(dissimilar_percentage))

        class_metrics = {
            "first": str(dissimilar_percentage1),
            "top5": str(dissimilar_percentage5),
            "topK": str(dissimilar_percentage)
        }

        if include_certainties:
            mean_cert1 = "0"
            mean_cert5 = "0"
            mean_cert = "0"
            stddev1 = "0"
            stddev5 = "0"
            stddev = "0"

            if (len(certainty_diff1) != 0):
                mean_cert1 = sum(certainty_diff1) / Decimal(len(certainty_diff1))
                squared_diffs1 = [(x - mean_cert1) ** 2 for x in certainty_diff1]
                variance1 = sum(squared_diffs1) / Decimal(len(certainty_diff1))
                stddev1 = variance1.sqrt()

            if (len(certainty_diff5) != 0):
                mean_cert5 = sum(certainty_diff5) / Decimal(len(certainty_diff5)) 
                squared_diffs5 = [(x - mean_cert5) ** 2 for x in certainty_diff5]
                variance5 = sum(squared_diffs5) / Decimal(len(certainty_diff5))
                stddev5 = variance5.sqrt()

            if (len(certainty_diff) != 0):
                mean_cert = sum(certainty_diff) / Decimal(len(certainty_diff))
                squared_diffs = [(x - mean_cert) ** 2 for x in certainty_diff]
                variance = sum(squared_diffs) / Decimal(len(certainty_diff))
                stddev = variance.sqrt()

            class_metrics["certainties"] = {
                "mean_top1": str(mean_cert1),
                "mean_top5": str(mean_cert5),
                "mean_topK": str(mean_cert),
                "stddev1": str(stddev1),
                "stddev5": str(stddev5),
                "stddevK": str(stddev)
            }
            
        self.comparisons[model_name][current_pass].update(class_metrics)
            

        if (dissimilar_percentage > 0):
            self.comparisons[model_name]["different"] += 1

        self.comparisons["no_dissimilar"] += 1 if dissimilar_percentage != 0 else 0
