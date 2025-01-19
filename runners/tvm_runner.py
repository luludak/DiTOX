
from runners.tvm_objects import ObjectsExecutor

class TVMRunner:

    def __init__(self, build_config, remote_config):
        self.build = build_config
        self.remote = remote_config

    def execute_tvm(self, models_data, images_data, specific_images, debug_enabled=False):
        models_data["debug_enabled"] = debug_enabled
        objectsExecutor = ObjectsExecutor(models_data, images_data, self.build)
        return objectsExecutor.execute(self.remote, specific_images)