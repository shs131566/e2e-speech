import tensorflow as tf

def bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = list_of_bytestrings))
    