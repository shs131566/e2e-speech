

from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.datasets.base_dataset import BaseDataset, BUFFER_SIZE, TFRECORD_SHARDS
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer


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