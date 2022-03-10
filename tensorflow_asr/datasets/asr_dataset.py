
import tensorflow as tf

from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.datasets.base_dataset import BaseDataset, BUFFER_SIZE, TFRECORD_SHARDS
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer

logger = tf.get_logger()

class ASRDataset(BaseDataset):
    def __init__(self, 
        stage:str, 
        speech_featurizer: SpeechFeaturizer,
        text_featurizer: TextFeaturizer,
        data_paths: list,
        augmentations: Augmentation = Augmentation(None),
        cache: bool = False, 
        shuffle: bool = False, 
        indefinite: bool = False, 
        drop_remainder: bool = True,
        use_tf: bool = False,
        buffer_size: int = BUFFER_SIZE,
        **kwargs
    ):
        super().__init__(
            data_paths = data_paths,
            augmentations = augmentations,
            cache = cache,
            shuffle = shuffle,
            stage = stage,
            buffer_size = buffer_size,
            drop_remainder = drop_remainder,
            use_tf = use_tf,
            indefinite = indefinite
        )
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    #! 임시코드
    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        print("TODO: ASRDataset -> parser is not implemented.")

    def create(self, batch_size: int):
        self.read_entries()

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0:
            return
        self.entries = []
        for file_path in self.data_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                 # Skip the header of tsv file
                self.entries += temp_lines[1:]
        self.entries = [line.split("\t", 2) for line in self.entries]
        for i, line in enumerate(self.entries):
            #self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1])])
            for x in self.text_featurizer.extract(line[-1]):
                print(str(x))
            #print(self.text_featurizer.extract(line[-1]))
            #self.text_featurizer.extract(line[-1])

class ASRTFRecordDataset(ASRDataset):
    def __init__(
        self,
        tfrecords_dir: str,
        speech_featurizer: SpeechFeaturizer,
        text_featurizer: TextFeaturizer,
        stage: str,
        augmentations: Augmentation = Augmentation(None),
        tfrecords_shards: int = TFRECORD_SHARDS,
        cache: bool = False,
        shuffle: bool = False,
        use_tf: bool = False,
        indefinite: bool = False,
        drop_remainder: bool = True,
        buffer_size: int = BUFFER_SIZE,
        **kwargs
    ):
        super().__init__(
            stage=stage,
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            data_paths=data_paths,
            augmentations=augmentations,
            cache=cache,
            shuffle=shuffle,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
            use_tf=use_tf,
            indefinite=indefinite,
        )
        if not self.stage:
            raise ValueError("stage must be defined, either 'train', 'eval' or 'test'")
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0:
            raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards

class ASRSliceDataset(ASRDataset):

    #! 임시 코드
    def dummy():
        raise NotImplementedError()