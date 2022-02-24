import logging
import warnings

import tensorflow as tf

from typing import List, Union

logger = tf.get_logger()

def setup_environment():
    warnings.simplefilter("ignore")
    logger.setLevel(logging.INFO)
    return logger

def setup_devices(devices: List[int], cpu: bool=False):
    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
        tf.config.set_visible_devices([], "GPU")
        logging.info(f"Run on {len(cpus)} Physical CPUs")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            visible_gpus = [gpus[i] for i in devices]
            tf.config.set_visible_devices(visible_gpus, "GPU")
            logger.info(f"Run on {len(visible_gpus)} Physical GPUs")
            
def setup_strategy(devices: List[int]):
    setup_devices(devices)
    return tf.distribute.MirroredStrategy()