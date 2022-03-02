import tensorflow as tf

from tensorflow_asr.augmentations.methods import specaugment

AUGMENTATIONS = {
    "freq_masking": specaugment.FreqMasking,
    "time_masking": specaugment.TimeMasking,
}

class Augmentation:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.prob = float(config.pop("prob", 0.5))
        self.signal_augmentations = self.parse(config.pop("signal_augment", {}))
        self.feature_augmentations = self.parse(config.pop("feature_augment", {}))

    @staticmethod
    def parse(config: dict) -> list:
        augmentations = []
        for k, v in config.items():
            aug = AUGMENTATIONS.get(k, None)
            if aug is None:
                raise KeyError(f"No tf augmentation named: {key}\n" f"Available tf augmentations: {AUGMENTATIONS.keys()}")
            aug = aug(**v) if v is not None else aug()
            augmentations.append(aug)
        return augmentations