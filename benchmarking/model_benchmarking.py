import onnx
import onnxruntime
from helpers.model_helper import get_size
import numpy as np
import time

def benchmark(model_path, config={}):
    file_size = get_size(model_path)
    print("File Size: ", file_size)
    onnx_model = onnx.load(model_path)
    session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider']) # , provider_options=[{"device_type": "GPU_FP32"}]
    print(onnxruntime.get_device())
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    shape = config["shape"] if "shape" in config else (1, 3, 224, 224)
    input_data = np.zeros(shape, np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        # print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")
    return total