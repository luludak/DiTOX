# DiTOX: Differential Testing of the ONNX Optimizer

DiTOX is a utility that enables differential testing of the ONNX Optimizer, by fetching automatically real-life ONNX models from the ONNX Model Hub and performing full and per-pass differential testing on the ONNX Optimizer.

![DiTOX](images/DiTOX.png)

```
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          25            897            472           2785
JSON                             3              4              0             84
Markdown                         1             17              0             59
-------------------------------------------------------------------------------
SUM:                            29            918            472           2928
-------------------------------------------------------------------------------
```

## Features
1. **Automatic Fetching and Loading of Models:** Fetch DNN models from model hubs (e.g., ONNX Model Hub) or load from local ONNX files.
2. **Support for Multiple Model Types:** Classifiers, object detectors, and transformers/text generators.
3. **Flexible Inference:** Run model inference on (a) a configuration-defined local dataset or (b) well-known datasets from the datasets package.
4. **Batch Model Analysis:** Analyze multiple models in fully automated batches.
5. **Model Optimization:** Optimize models using the ONNX optimizer, either fully (all passes) or partially (selected passes). Users can also define custom pass orders.
6. **Automatic Model Evaluation:** Evaluate models using appropriate metrics: Kendall Tau (classifiers), F1/Precision/Recall (object detectors), BLEU (transformers), etc.
7. **Per-Pass Fault Localization:** If full optimization causes model corruption, crashes, or accuracy drops, DiTOX generates per-pass variants and performs differential testing to identify problematic passes.
8. **Detailed Report Generation:** Generate reports containing extensive metadata, including metric values at different thresholds and class certainties.
9. **Automatic Logging of Bugs:** Log detected issues automatically without interrupting the analysis.
10. **Chunk-Based Execution:** Split datasets into batches for resource-intensive models, generating partial results for each run.

## Effectiveness
DiTOX has discovered `15` bugs. Of these, `14` were entirely new (not previously reported on the ONNX Optimizer issue tracker). The issues were associated with `9` of the `47` passes in the optimizer. We have reported the issues to the ONNX Optimizer, while we also contain the raw results for all models and the bugs detected in the repo.

## Installation/Running
1. Install necessary packages by doing `pip install -r requirements.txt`.
2. Verify that the ONNX Optimizer is properly installed.
3. Adjust configuration in `config.json`
4. Run by doing `python main.py`

By this setting, DiTOX will load the respective configuration and perform the necessary runs for the batch of models selected.

## Configuration
- To use local image datasets to examine vision models, set the path in the `images_folder_rel_path` variable. Specify its start sample index on `starts_from` and its size on `limit` parameters of the `images` object.
- If you wish to chunk the experiments, set the `images_chunk` size to your desired chunk size, otherwise set the same value as `limit`.
- The parameter of `run_individual_passes` will run the system by optimizing each model to either all `Fuse/Eliminate` passes (if set to `true`), otherwise to each pass in isolation.
You can also set it to true and define the desired passes to run by setting them up in `optimizer/passes` parameter. You can find the full list of pass names in `pass.txt`.
- You can define the ONNX hub settings and filters to fetch the appropriate models in the `hub` object parameters.
- You can also filter the models you want to use by restricting the id range of models considered in `skip_until_model_id` and `run_up_to_model_id`, as well as set name filters for the models using the `name_filters` parameter, under `general` object.
- If you enable `check_reports_for_problematic_models`, DiTOX will check for problematic models in a file with name set in `report_file_base_rel_path`. If a model is marked as "different" in this file, it will then perform per-pass fault localization + differential testing for this model, by applying each available pass, sequentially.
- `include_certainties` is an experimental feature that includes certainty percentages. It is currently supported for classification models.
- `enable_kt_on_one_dim_tensor` is another experimental feature that allows for Kendall Tau Correlation Coefficient to be enabled or disabled for (batch, dim) tensors.

## Demonstration
For demonstration purposes, we have included 10 images and set the configuration to fetch and run all `YOLO` (object detection) and `MobileNet` (classification) models from ONNX hub.

When run, DiTOX will generate an output file based on the value set in the `general/report_file_base_rel_path` parameter. However, DiTOX will split the first `_chunk` encountered and replace it with `<starts_from>_<limit>.json` based on the configuration. If the experiments are chunked, it will generate multiple files, each with its respective range. If you replace your base file, make sure it ends with `_chunk`.


## Benchmarking/Issues Replication
By opening configuration file and setting the model of each encountered error in model name filters as well the problematic pass in the `passes` you should be able to replicate the issue reported.

## Usage
The system utilizes a configuration file in order to define the dataset path, but also filter the models.
By default, it fetches the models from the ONNX Model Hub. The models can be selected by ID range (so that a subset of models can be executed in the system in line), as well as by type (e.g., vision) and name (e.g., containing the keyword "YOLO", to extract all YOLO models) The provided name can be matched as an exact term (by setting `match_filter_exact` to true), or a term being part of the model name and opset otherwise.
Note that, in order to perform exact matching, you must use the form `<Model-Name-As-In-Model-Hub>-<opset>`.

For a sample run, the repo contains 10 images obtained from the internet in general, provided *only* for demonstration purposes of the project.
If you want to use your own dataset, you can define the path in the configuration file. You can also chunk the runs if the dataset is too large, using the `images/images_chunk` property in configuration, while you can also define the start and the end of images in the dataset (`images/starts_from` and `images/limit` properties). This is handy in case an experiment failed, and you want to continue from this point.

You can also instruct DiTOX to run the primary passes for each model, or all passes step-by-step, via the `general/run_all_passes` property in configuration file.
Finally, you can select individual passes to be run, by setting their names in the `optimizer/passes` property in configuration file.

## Notes
`main.py` contains the code for the experiments for the classification and the object detection models.
The sample code related to the text generation models is contained in `main-GPT2-Complete.py`, which you can also run.
Essentially the same code (with minor changes) is used to test RoBERTa, BERT-Squad, and T5 - with the token processing policy and the comparator settings changing slightly.

