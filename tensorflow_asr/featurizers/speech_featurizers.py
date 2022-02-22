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
    # audio 파일 경로가 입렵된 경우
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=None, mono=True)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
