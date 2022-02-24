import tensorflow as tf

from tensorflow_asr.augmentations.methods import specaugment

AUGMENTATIONS = {
    "freq_masking": specaugment.FreqMasking,
    "time_masking": specaugment.TimeMasking,
}