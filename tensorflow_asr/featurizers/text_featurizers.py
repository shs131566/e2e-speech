import re
import abc
import hgtk
import codecs
import unicodedata

import tensorflow as tf

from tensorflow_asr.configs.config import DecoderConfig

ENGLISH_CHARACTERS = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", 
                      "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                      "T", "U", "V", "W", "X", "Y", "Z", "'"]

logger = tf.get_logger()

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
        text = unicodedata.normalize("NFC", text.upper())
        return text.strip("\n")

    @abc.abstractclassmethod
    def extract(self, text):
        raise NotImplementedError()

class SentencePieceFeaturzier(TextFeaturizer):
    UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 1
    BOS_TOKEN, BOS_TOKEN_ID = "<s>", 2
    EOS_TOKEN, EOS_TOKEN_ID = "</s>", 3
    PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 0

    #! 임시 코드 
    def __init__(self, decoder_config: dict, model=None):
        print("TODO: SentencePieceFeaturzier -> __init__ is not implemented.")

class SubwordFeaturizer(TextFeaturizer):
    #! 임시 코드
    def __init__(self, decoder_config: dict, subwords=None):
        print("TODO: SubwordFeaturizer -> __init__ is not implemented.")

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

    def extract(self, text: str) -> tf.Tensor:
        text = self.preprocess_text(text)
        text = list(text.strip())
        indices = [self.tokens2indices[token] for token in text]
        return tf.convert_to_tensor(indices, dtype=tf.int32)

class KoreanGraphemeFeaturizer(TextFeaturizer):
    KOREAN = ["SPACE", "CHO", "JOONG", "JONG"]
    KOREAN_GRAPHEMES = [' ', list(hgtk.const.CHO), list(hgtk.const.JOONG), list(hgtk.const.JONG)[1:]]

    def __init__(self, decoder_config: dict):
        super(KoreanGraphemeFeaturizer, self).__init__(decoder_config)
        self.__init_vocabulary()
        print(self.tokens2indices)

    def __init_vocabulary(self):
        lines = []
        if self.decoder_config.vocabulary is not None:
            with codecs.open(self.decoder_config.vocabulary, "r", "utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            lines = self.KOREAN_GRAPHEMES
        self.blank = 0 if self.decoder_config.blank_at_zero else None
        self.tokens2indices = {}
        self.tokens = []
        index = 1 if self.blank == 0 else 0
        for idx in self.KOREAN:
            self.tokens2indices[idx] = {}
            for grapheme in lines[self.KOREAN.index(idx)]:
                self.tokens2indices[idx][grapheme] = index
                self.tokens.append(grapheme)
                index += 1
        if self.blank is None:
            self.blank = len(self.tokens)
        self.non_blank_tokens = self.tokens.copy()
        self.tokens.insert(self.blank, "")
        self.num_classes = len(self.tokens)
        self.tokens = tf.convert_to_tensor(self.tokens, dtype=tf.string)
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(shape=[None, 1])
        self.hangul = re.compile("[^ 가-힣]+")

    # TODO: 입력 텍스트 정규표현식으로 
    def extract(self, text):
        if self.hangul.findall(text) != []:
            logger.warning(f"{text} isn't written in Korean. {self.hangul.findall(text)} will be ignored.")
        text = self.hangul.sub('', text) 
        text = hgtk.text.decompose(text.strip(), compose_code="*")
        text = self.preprocess_text(text)
        text = text.split("*")
        indices = []
        for character in text:
            idx = 1
            for grapheme in character:
                if grapheme is ' ':
                    idx -= 1        
                indices.append(self.tokens2indices[self.KOREAN[idx]][grapheme])
                idx += 1
        print([g for c in text for g in c])
        print(indices)
        return tf.convert_to_tensor(indices, dtype=tf.int32)