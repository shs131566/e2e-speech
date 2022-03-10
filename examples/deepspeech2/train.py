import os
import argparse

import tensorflow as tf

from tensorflow_asr.utils import env_util
from tensorflow_asr.datasets import asr_dataset
from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers import speech_featurizers, text_featurizers

logger = env_util.setup_environment()

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="DeepSpeech2 Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords")
parser.add_argument("--korean", default=False, action="store_true", help="Whether to make Korean model.")
parser.add_argument("--sentence_piece", default=False, action="store_true", help="whether to use 'SentencePiece' model")
parser.add_argument("--subwords", default=False, action="store_true", help="Use subwords")
parser.add_argument("--bs", type=int, default=None, help="Batch size per replica")
parser.add_argument("--spx", type=int, default=1, help="Steps per execution for maximizing performance")
parser.add_argument("--static_length", default=False, action="store_true", help="Use static lengths")
parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")
parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")
parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precsion": args.mxp})

startegy = env_util.setup_strategy(args.devices)

config = Config(args.config)

speech_featurizer = speech_featurizers.TFSpeechFeaturizer(config.speech_config)

if args.korean:
    logger.info("Use Korean grapheme ...")
    text_featurizer = text_featurizers.KoreanGraphemeFeaturizer(config.decoder_config)

else:
    if args.sentence_piece:
        logger.info("Loading English SentencePiece model ...")
        text_featurizer = text_featurizers.SentencePieceFeaturzier(config.decoder_config)
    elif args.subwords:
        logger.info("Loading English subwords ...")
        text_featurizer = text_featurizers.SubwordFeaturizer(config.decoder_config)
    else:
        logger.info("Use English characters ...")
        text_featurizer = text_featurizers.CharFeaturizer(config.decoder_config)

if args.tfrecords:
    train_dataset = asr_dataset.ASRTFRecordDataset(
        speech_featurizer = speech_featurizer,
        text_featurizer = text_featurizer,
        **vars(config.learning_config.train_dataset_config),
        indefinite = True
    )
    eval_dataset = asr_dataset.ASRTFRecordDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.eval_dataset_config),
        indefinite=True
    )
else:
    train_dataset = asr_dataset.ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.train_dataset_config),
        indefinite=True
    )
    eval_dataset = asr_dataset.ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.eval_dataset_config),
        indefinite=True
    )

global_batch_size = args.bs or config.learning_config.running_config.batch_size
global_batch_size *= startegy.num_replicas_in_sync

train_data_loader = train_dataset.create(global_batch_size)
