import os
import io
import librosa

import numpy as np
import soundfile as sf
import tensorflow as tf

from typing import Union

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
