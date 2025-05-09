# DiTOX: Differential Testing of the ONNX Optimizer

DiTOX is a utility that enables differential testing of the ONNX Optimizer, by fetching automatically real-life ONNX models from the ONNX Model Hub and performing full and per-pass differential testing on the ONNX Optimizer.

![DiTOX](https://github.com/user-attachments/assets/8a993198-bec0-446d-9760-9eb6aa1f20ed)


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

## Note
`main.py` contains the code for the experiments for the classification and the object detection models. The code related to the text generation models is contained in `main-GPT2-Complete.py` but is in draft state. It will be refactored soon and polished before publication.
