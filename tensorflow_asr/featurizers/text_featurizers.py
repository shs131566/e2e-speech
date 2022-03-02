import abc
import unicodedata

import tensorflow as tf

from tensorflow_asr.configs.config import DecoderConfig

ENGLISH_CHARACTERS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", 
                      "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                      "t", "u", "v", "w", "x", "y", "z", "'"]

class TextFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self, decoder_config: dict):
        self.scorer = None
        self.decoder_config = DecoderConfig(decoder_config)
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None
        self.max_length = 0

    def preprocess_text(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        return text.strip("\n")

class SentencePieceFeaturzier(TextFeaturizer):
    UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 1
    BOS_TOKEN, BOS_TOKEN_ID = "<s>", 2
    EOS_TOKEN, EOS_TOKEN_ID = "</s>", 3
    PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 0

    #! 임시 코드 
    def __init__(self, decoder_config: dict, model=None):
        return None

class SubwordFeaturizer(TextFeaturizer):
    #! 임시 코드
    def __init__(self, decoder_config: dict, subwords=None):
        return None

class CharFeaturizer(TextFeaturizer):
    def __init__(self, decoder_config: dict):
        super(CharFeaturizer, self).__init__(decoder_config)
        self.__init_vocabulary()
    
    def __init_vocabulary(self):
        lines = []
        if self.decoder_config.vocabulary is not None:
            with codecs.open(self.decoder_config.vocabulary, "r", "utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            lines = ENGLISH_CHARACTERS
        self.blank = 0 if self.decoder_config.blank_at_zero else None
        self.tokens2indices = {}
        self.tokens = []
        index = 1 if self.blank == 0 else 0
        for line in lines:
            line = self.preprocess_text(line)
            if line.startswith("#") or not line:
                continue
            self.tokens2indices[line[0]] = index
            self.tokens.append(line[0])
            index += 1
        if self.blank is None:
            self.blank = len(self.tokens)
        self.non_blank_tokens = self.tokens.copy()
        self.tokens.insert(self.blank, "")
        self.num_classes = len(self.tokens)
        self.tokens = tf.convert_to_tensor(self.tokens, dtype=tf.string)
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(shape=[None, 1])
            