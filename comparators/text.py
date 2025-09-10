class TextComparator:

    def __init__(self, evaluation, comparisons):
        self.evaluation = evaluation
        self.comparisons = comparisons

    def update(self, model_name, current_pass="all", include_certainties=False):
        dissimilar_percentage1 = self.evaluation["percentage_dissimilar1"]
        dissimilar_percentage5 = self.evaluation["percentage_dissimilar5"]
        dissimilar_percentage = self.evaluation["percentage_dissimilar"]
        
        print("Dissimilarity for " + current_pass + " (top-1): " + str(dissimilar_percentage1))
        print("Dissimilarity for " + current_pass + " (top-5): " + str(dissimilar_percentage5))
        print("Dissimilarity for " + current_pass + " (top-K): " + str(dissimilar_percentage))

        self.comparisons[model_name][current_pass] = {
            "first": str(dissimilar_percentage1),
            "top5": str(dissimilar_percentage5),
            "topK": str(dissimilar_percentage)
        }

        if (dissimilar_percentage > 0):
            self.comparisons[model_name]["different"] += 1

        self.comparisons["no_dissimilar"] += 1 if dissimilar_percentage != 0 else 0
