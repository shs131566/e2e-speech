from codecs import ignore_errors
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.utils import math_util, feature_util, data_util
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.datasets.base_dataset import BaseDataset, BUFFER_SIZE, TFRECORD_SHARDS, AUTOTUNE
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer, tf_read_raw_audio

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
            self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1]).numpy()])
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)
        self.total_steps = len(self.entries)

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        raise NotImplementedError()

    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):

        data = self.tf_preprocess(path, audio, indices) if self.use_tf else self.preprocess(path, audio, indices)
        raise NotImplementedError()

    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)
            # signal = self.augmentations.signal_augmentations(signal)
            # features = self.speech_featurizer.tf_extract(signal)
            # TODO: tf_extract(...)
        raise NotImplementedError()

class ASRTFRecordDataset(ASRDataset):
    """ Dataset for ASR using TFRecords """
    def __init__(
        self,
        data_paths: list,
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
        **kwargs,
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

    def parse(self, record: tf.Tensor):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        tf.print(record)
        exit(1)
        return super().parse(**example)
    
    def create(self, batch_size: int):
        have_data = self.create_tfrecords()
        if not have_data: return None
        
        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type="ZLIB", num_parallel_reads=AUTOTUNE)

        return self.process(dataset, batch_size)
    
    def create_tfrecords(self):
        if not self.tfrecords_dir: return False

        if tf.io.gfile.glob(os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecords")):
            logger.info(f"TFRecords're already existed: {self.stage}")
            return True
        
        logger.info(f"Creating {self.stage}.tfrecord ...")

        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return False

        def get_shard_path(shard_id):
            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")
        
        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_entries = np.array_split(self.entries, self.tfrecords_shards)
        for entries in zip(shards, splitted_entries):
            self.write_tfrecord_file(entries)
        
        return True

    def write_tfrecord_file(self, splitted_entries):
        shard_path, entries = splitted_entries

        # def parse(record):
        #     def fn(path, indices):
        #         audio = load_and_convert_to_wav(path.decode("utf-8")).numpy()
        #         feature = {
        #             "path": feature_util.bytestring_feature([path]),
        #             "audio": feature_util.bytestring_feature([audio]),
        #             "indices": feature_util.bytestring_feature([indices])
        #         }
        #         print(feature)
        
        dataset = tf.data.Dataset.from_tensor_slices(entries)