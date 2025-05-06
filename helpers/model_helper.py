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

def get_model_path(script_dir, model_obj):
    return script_dir + "/models_cache/" + model_obj.model_path.replace("/model/", "/model/" + \
            model_obj.metadata["model_sha"] + "_").replace("/models/", "/models/" + \
            model_obj.metadata["model_sha"] + "_").replace("/preproc/", "/preproc/" + \
            model_obj.metadata["model_sha"] + "_")

def prepare_model_shape(model_obj):
    # Default shape definition.
    model_shape = [1, 1, 64, 64]
    first_input = model_obj.metadata["io_ports"]["inputs"][0]
    input_name = None
    model_name = model_obj.model

    if "io_ports" in model_obj.metadata:
        first_input = model_obj.metadata["io_ports"]["inputs"][0]
        model_shape = first_input["shape"]
    
    # Consider corner cases for R-CNN and FCN models.
    if "R-CNN" in model_name:
        model_shape = [1, 3, 224, 224]
    elif "FCN" in model_name or "YOLOv3" in model_name:
        model_shape = [1, 3, 224, 224]
    return model_shape


def prepare_model_input_dimension(model_obj):
    

    m_shape = prepare_model_shape(model_obj).copy()
    if (len(m_shape) > 0):
        m_shape.pop(0)

    img_dimension = []
    for img_dim in m_shape:
        if type(img_dim) == str:
            continue
        elif (img_dim > 5):
            img_dimension.append(img_dim)

    print(img_dimension)
    return img_dimension