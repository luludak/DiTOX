# DiTOX: Differential Testing of the ONNX Optimizer

DiTOX is a utility that enables differential testing of the ONNX Optimizer, by fetching automatically real-life ONNX models from the ONNX Model Hub and performing full and per-pass differential testing on the ONNX Optimizer.

![DiTOX](images/DiTOX.png)

```
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          22            866            568           2551
JSON                             1              1              0             25
Markdown                         1              8              0             22
-------------------------------------------------------------------------------
SUM:                            24            875            568           2598
-------------------------------------------------------------------------------
```


## Effectiveness
DiTOX has discovered `15` bugs. Of these, 14 were entirely new (not previously reported on the ONNX Optimizer issue tracker). The issues were associated with `9` of the `47` passes in the optimizer. We have reported the issues to the ONNX Optimizer, while we also contain the raw results for all models and the bugs detected in the repo.

## Installation
1. Install necessary packages by doing `pip install -r requirements.txt`.
2. Verify that the ONNX Optimizer is properly installed.
3. Adjust configuration in `config.json`
4. Run by doing `python main.py`

## Usage
The system utilizes a configuration file in order to define the dataset path, but also filter the models.
By default, it fetches the models from the ONNX Model Hub. The models can be selected by ID range (so that a subset of models can be executed in the system in line), as well as by type (e.g., vision) and name (e.g., containing the keyword "YOLO", to extract all YOLO models).

For a sample run, the repo contains 10 images from the ILSVRC dataset, used for demonstration purposes.
If you want to use your own dataset, you can define the path in the configuration file. You can also chunk the runs if the dataset is too large, using the `images/images_chunk` property in configuration, while you can also define the start and the end of images in the dataset (`images/starts_from` and `images/limit` properties). This is handy in case an experiment failed, and you want to continue from this point.

You can also instruct DiTOX to run the primary passes for each model, or all passes step-by-step, via the `general/run_all_passes` property in configuration file.
Finally, you can select individual passes to be run, by setting their names in the `optimizer/passes` property in configuration file.

## Notes
`main.py` contains the code for the experiments for the classification and the object detection models.
The sample code related to the text generation models is contained in `main-GPT2-Complete.py` but is in draft state.
Essentially the same code (with minor changes) is used to test RoBERTa, BERT-Squad, and T5 - with the token processing policy and the comparator settings
changing slightly. We aim to fully polish and release it when we make the project public.
