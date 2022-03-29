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
                raise KeyError(f"No tf augmentation named: {k}\n" f"Available tf augmentations: {AUGMENTATIONS.keys()}")
            aug = aug(**v) if v is not None else aug()
            augmentations.append(aug)
        return augmentations

    @tf.function
    def signal_augment(self, inputs):
        return self._augment(inputs, self.signal_augmentations)

    def _augment(self, inputs, augmentations):
        outputs = inputs
        for au in augmentations:
            p = tf.random.uniform([])
            output = tf.where(tf.less(p, self.prob), au.augment(outputs), outputs)
        return outputs