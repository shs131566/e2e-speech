import math

def get_num_batches(nsamples, batch_size, drop_remainder = True):
    if nsamples is None or batch_size is None:
        return None
    if drop_remainder:
        return math.floor(float(nsamples) / float(batch_size))
    return math.ceil(float(nsamples) / float(batch_size))