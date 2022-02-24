import os
import argparse

import tensorflow as tf

from tensorflow_asr.utils import env_util
from tensorflow_asr.configs.config import Config

logger = env_util.setup_environment()

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="DeepSpeech2 Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords")
parser.add_argument("--sentence_piece", default=False, action="store_true", help="whether to use 'SentencePiece' model")
parser.add_argument("--subwords", default=False, action="store_true", help="Use subwords")
parser.add_argument("--bs", type=int, default=None, help="Batch size per replica")
parser.add_argument("--spx", type=int, default=1, help="Steps per execution for maximizing performance")
parser.add_argument("--metadata", type=str, default=None, help="Path to file containing metadata")
parser.add_argument("--static_length", default=False, action="store_true", help="Use static lengths")
parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")
parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")
parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precsion": args.mxp})

startegy = env_util.setup_strategy(args.devices)

config = Config(args.config)
