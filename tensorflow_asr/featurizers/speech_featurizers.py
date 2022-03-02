import os
import io
import abc
import librosa

import numpy as np
import soundfile as sf
import tensorflow as tf

from typing import Union

class SpeechFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self, speech_config: dict):
        self.sample_rate = speech_config.get("sample_rate", 16000)
        self.frame_length = int(self.sample_rate * (speech_config.get("frame_ms", 25) / 1000))
        self.frame_step = int(self.sample_rate * (speech_config.get("stride_ms", 10) / 1000))

        self.num_feature_bins = speech_config.get("num_feature_bins", 80)
        self.feature_type = speech_config.get("feature_type", "log_mel_spectrogram")
        self.preemphasis = speech_config.get("preemphasis", None)
        self.top_db = speech_config.get("top_db", 80.0)

        self.normalize_signal = speech_config.get("normalize_signal", True)
        self.normalize_feature = speech_config.get("normalize_feature", True)
        self.normalize_per_frame = speech_config.get("normalize_per_frame", False)
        self.center = speech_config.get("center", True)

        self.max_length = 0

class TFSpeechFeaturizer(SpeechFeaturizer):
    def dummy():
        return None

def load_and_convert_to_wav(path: str) -> tf.Tensor:
    wave, rate = librosa.load(os.path.expanduser(path), sr=None, mono=True)
    return tf.audio.encode_wav(tf.expand_dims(wave, axis=1), sample_rate=rate)

def read_raw_audio(audio: Union[str, bytes, np.ndarray], sample_rate=16000) -> np.ndarray:
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1:
            wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        if audio.ndim > 1:
            ValueError("input audio must be single channel")
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return audio

def tf_read_raw_audio(audio: tf.Tensor, sample_rate=16000) -> tf.Tensor:
    return None
