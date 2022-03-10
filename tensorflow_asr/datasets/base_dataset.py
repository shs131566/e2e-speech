import abc

from tensorflow_asr.augmentations.augmentation import Augmentation

BUFFER_SIZE = 100
TFRECORD_SHARDS = 16

class BaseDataset(metaclass=abc.ABCMeta):
    def __init__(
        self,
        data_paths: list,
        augmentations: Augmentation = Augmentation(None),
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        use_tf: bool = False,
        stage: str = "train",
        **kwargs
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError("data_paths must be a list of string paths")
        self.augmentations = augmentations
        self.cache = cache
        self.shuffle = shuffle
        if buffer_size <= 0 and shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")
        self.buffer_size = buffer_size
        self.stage = stage
        self.use_tf = use_tf
        self.drop_remainder = drop_remainder
        self.indefinite = indefinite
        self.total_steps = None

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError()