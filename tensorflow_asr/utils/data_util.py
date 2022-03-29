import tensorflow as tf

def create_inputs(
    inputs: tf.Tensor,
    inputs_length: tf.Tensor,
    predictions: tf.Tensor = None,
    predictions_length: tf.Tensor = None,
) -> dict:
    data = {
        "inputs": inputs,
        "inputs_length": inputs_length,
    }
    if predictions is not None:
        data["predictions"] = predictions
    if predictions_length is not None:
        data["predictions_length"] = predictions_length
    return data
