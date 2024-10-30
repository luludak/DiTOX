import os
import json

def get_size(model_path):
    size = os.path.getsize(model_path)
    if size < 1024:
        return f"{size} bytes"
    elif size < pow(1024,2):
        return f"{round(size/1024, 2)} KB"
    elif size < pow(1024,3):
        return f"{round(size/(pow(1024,2)), 2)} MB"
    elif size < pow(1024,4):
        return f"{round(size/(pow(1024,3)), 2)} GB"

def load_config(model_path):
    config_path = model_path.replace(".onnx", ".json")
    if not os.path.exists(config_path):
        print("Warning: config for " + model_path + " was not found.\nUsing default config...")
        return {}
    json_data = open(config_path, "r")
    config = json.load(json_data)
    return config
