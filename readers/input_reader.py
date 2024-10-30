import numpy
import onnxruntime
import os
import onnx
from onnxruntime.quantization import CalibrationDataReader
from processors.model_preprocessor import ModelPreprocessor
from PIL import Image


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0, shape=(1, 3, 224, 224), model_name=""):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))

        preprocessing_data = {
            "input": shape,
            "input_dimension": (width, height),
            "library": None
        }
        # Set library to None, so that the name is utilized for preprocessing selection.
        model_preprocessor = ModelPreprocessor(preprocessing_data)
        pillow_img = model_preprocessor.preprocess(model_name, pillow_img, True)
        # Preprocess image using preprocessor.
        input_data = numpy.float32(pillow_img)

        if (input_data.shape[-1] >= input_data.shape[-2]):
            nchw_data = input_data
        else:
            nchw_data = input_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class InputReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, model_name: str):
        self.enum_data = None
        self.model_name = model_name

        # Use inference session to get input shape.
        onnx_model = onnx.load(model_path)
        session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider']) # , provider_options=[{"device_type": "GPU_FP32"}]
        # TODO: Add condition for dimensions
        shape = session.get_inputs()[0].shape
        print(shape)

        if shape[-1] < shape[-2]:
            (_, height, width, _) = shape
        else:
            (_, _, height, width) = shape


        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0, shape=shape, model_name=self.model_name
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)
        print(self.datasize)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None