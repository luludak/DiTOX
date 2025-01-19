import sys
sys.path.append("..")

from .tvm_executor import Executor

import os

class ObjectsExecutor(Executor):
    def __init__(self, models_data, images_data, connection_data):
        Executor.__init__(self, models_data, images_data, connection_data, "tvm_out")

    def execute(self, remote, specific_images=None):
        device_id = self.connection_data["id"] if "id" in self.connection_data else 0
        timestamp_str = str(self.get_epoch_timestamp()) + "_" + self.connection_data["device_name"] + str(device_id)

        start_timestamp = self.get_epoch_timestamp(False)
        self.prepare(remote=remote)

        print("Executing model " + self.name + ", execution timestamp: " + timestamp_str)
    
        output_obj = self.process_images_with_io(self.input_images_folders[0], self.output_model_folder, self.model_name, "", specific_images, should_write_to_file=False)
        end_timestamp = self.get_epoch_timestamp(False)

        return output_obj
