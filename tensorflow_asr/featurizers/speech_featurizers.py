import os
import io
import abc
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
    wave, rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
    return tf.reshape(wave, shape=[-1])

def tf_normalize_signal(signal: tf.Tensor) -> tf.Tensor:
    gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
    return signal * gain

def tf_preemphasis(signal: tf.Tensor, coeff = 0.97):
    if not coeff or coeff <= 0.0:
        return signal
    s0 = tf.expand_dims(signal[0], axis=-1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], axis=-1)


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
    # TODO: ast_dataset.py -> ASRDataset class : tf_preprocess
    def tf_extract(self, signal: tf.Tensor) -> tf.Tensor:
        if self.normalize_signal:
            signal = tf_normalize_signal(signal)
        signal = tf_preemphasis(signal, self.preemphasis)

        if self.feature_type == "spectrogram":
            features = self.compute_spectrogram(signal)
            # TODO: 작성중 ...
        raise NotImplementedError()

    #TODO: speech_featurizers.py -> TFSpeechFeaturizer class : tf_extract
    def compute_spectrogram(self, signal: tf.Tensor) -> tf.Tensor:
        S = self.sftf(signal)
        # TODO: 작성중...
        raise NotImplementedError()

    #TODO: speech_featurizers.py -> TFSpeechFeaturizer class : compute_spectrogram
    def stft(self, signal: tf.Tensor) -> tf.Tensor:
        if self.center:
            signal = tf.pad(signal, [[self.nfft // 2, self.nfft // 2]], mode = "REFLECT")
        window = tf.signal.hann_window(self.frame_length, periodic = True)
        left_pad = (self.nfft - self.frame_length) // 2
        right_pad = self.nfft - self.frame_length - left_pad
        window = tf.pad(window, [[left_pad, right_pad]])
        framed_signals = tf.signal.frame(signal, frame_length=self.nfft, frame_step=self.frame_step)
        frame_signals *= window
        return tf.square(tf.abs(tf.signal.rfft(framed_signals, [self.nfft])))
