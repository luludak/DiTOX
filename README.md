# DiTOX: Differential Testing of the ONNX Optimizer

DiTOX is a system that allows differential testing of the ONNX Optimizer, by fetching automatically real-life ONNX models from the ONNX Model Hub
and performing full and per-pass differential testing on the ONNX Optimizer. DiTOX has discovered `15` bugs, of which `14` are new.

Note: `main.py` contains the code for the experiments for the classification and the object detection models.
The code related to the text generation models is contained in `main-GPT2-Complete.py` but is in draft state.
It will be refactored soon and polished before publication.
