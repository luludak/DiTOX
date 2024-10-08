import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
import tensorflow as tf
import glob
import os

from tensorflow import keras

script_dir = os.path.dirname(os.path.realpath(__file__))
dataset_list = [f for f in glob.glob(script_dir + "/images/small/*")]
step = len(dataset_list) // 4

def representative_data_gen():
    imgs_proc = 0
    for image_file in dataset_list:
        print(image_file)
        if(imgs_proc % step == 0):
            print("Complete: " + str((imgs_proc // step) * 25) + "%")

        imgs_proc += 1
        
        image = tf.io.read_file(image_file)

        if image_file.endswith('.jpg') or image_file.endswith('.JPG') or image_file.endswith('.JPEG'):
            image = tf.io.decode_jpeg(image, channels=3)
        elif image_file.endswith('.png'):
            image = tf.io.decode_png(image, channels=3)
        elif image_file.endswith('.bmp'):
            image = tf.io.decode_bmp(image, channels=3)

        image = tf.image.resize(image, [224, 224])  
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        
        yield [image]
        
        
print(representative_data_gen())
    
# data = {
#     "input_name": "Placeholder",
#     "output_names": ["final_dense"],
#     "input": [1, 3, 224, 224]
# }

# PB_PATH = script_dir + "/models/densenet.pb"
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(PB_PATH, #TensorFlow freezegraph,
#                                                 input_shapes={data["input_name"]: data["input"]},
#                                                 input_arrays=[data["input_name"]], # name of input
#                                                 output_arrays=data["output_names"]  # name of output
#                                                 )
converter = tf.lite.TFLiteConverter.from_keras_model(keras.applications.MobileNetV2(include_top=True, weights="imagenet"))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

tf_lite_model = converter.convert()
TFLITE_PATH = script_dir + "/models/MobileNetV2.tflite"
open(TFLITE_PATH, 'wb').write(tf_lite_model)


#converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(script_dir + "/models/mobilenet_v2.tflite")
debugger = tf.lite.experimental.QuantizationDebugger(quant_debug_model_path=TFLITE_PATH, converter=converter, debug_dataset=representative_data_gen)
debugger.run()


RESULTS_FILE = script_dir + '/debugger_results_mobilenetv2_k_tflite_new.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)

layer_stats = pd.read_csv(RESULTS_FILE)
layer_stats.head()

layer_stats['range'] = 255.0 * layer_stats['scale']
layer_stats['rmse/scale'] = layer_stats.apply(
    lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)


layer_data = []
ind = 0
for i in range(len(layer_stats['rmse/scale'])):
    tensor_name = layer_stats['tensor_name'][i]
    id = layer_stats['tensor_idx'][i]
    scale = layer_stats['rmse/scale'][i]
    op_name = layer_stats['op_name'][i]
    if "CONV_2D" in op_name:
        layer_data.append({"i": ind, "idx": i, "name": tensor_name, "scale": scale, "op_name": op_name})
        ind += 1

layer_data.sort(key=lambda x: x["scale"], reverse=True)

# ids = [(x["i"], x["name"], x["scale"]) for x in layer_data]
ids = [x["i"] for x in layer_data] #  if x["scale"] > 0.289
print(ids)
print(len(ids))
