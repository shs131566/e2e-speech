from typing import Union

from tensorflow_asr.utils import file_util
from tensorflow_asr.augmentations.augmentation import Augmentation

class Config:
    def __init__(self, data: Union[str, dict]):
        config = data if isinstance(data, dict) else file_util.load_yaml(file_util.preprocess_paths(data))
        self.speech_config = config.pop("speech_config", {})
        self.decoder_config = config.pop("decoder_config", {})
        self.model_config = config.pop("model_config", {})
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        print(config)
        for k, v in config.items():
            setattr(self, k, v)

class LearningConfig:
    def __init__(self, config: dict=None):
        if not config:
            config = {}
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_config = DatasetConfig(config.pop("test_dataset_config", {}))
        self.optimizer_config = config.pop("optimizer_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        for k, v in config.items():
            setattr(self, k, v)

class DatasetConfig:
    def __init__(self, config: dict=None):
        if not config:
            config = {}
        self.stage = config.pop("stage", None)
        self.data_paths = file_util.preprocess_paths(config.pop("data_paths", None))
        self.tfrecords_dir = file_util.preprocess_paths(config.pop("tfrecords_dir", None), isdir=True)
        self.tfrecords_shards = config.pop("tfrecords_shards", 16)
        self.shuffle = config.pop("shuffle", False)
        self.cache = config.pop("cache", False)
        self.drop_remainder = config.pop("drop_remainder", True)
        self.buffer_size = config.pop("buffer_size", 100)
        self.use_tf = config.pop("use_tf", False)
        self.augmentations = Augmentation(config.pop("augmentation_config", {}))
        for k, v in config.items():
            setattr(self, k, v)