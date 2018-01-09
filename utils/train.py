import tensorflow as tf
import tensorpack.dataflow as df
import dataflow
import dataflow.tensorflow

import logging
logger = logging.getLogger(__name__)


def build_remote_feeder_thread(port, batchsize): 
    ds = df.RemoteDataZMQ('tcp://0.0.0.0:' + str(port))
    ds = df.PrefetchDataZMQ(ds, nr_proc=1)
    ds.reset_state()

    # feed data queue input
    with tf.device(tf.DeviceSpec(device_type='CPU', device_index=0)):
        _placeholders = [
            tf.placeholder(tf.float32, (batchsize, 224, 224, 3)),
            tf.placeholder(tf.int64, (batchsize,))
        ]
        thread = dataflow.tensorflow.QueueInput(
            ds, _placeholders, repeat_infinite=False, queue_size=50)
    return thread

def build_learning_rate(global_step, device_count, params):
    return _build_learning_rate(global_step, device_count, params['steps_per_epoch'], params['batchsize'], params['learning_rate']['initial'], params['learning_rate']['warmup'])

def _build_learning_rate(global_step, device_count, steps_per_epoch, batchsize, initial_learning_rate, warmup=True):
    multiplier = (batchsize * device_count) / 256.0
    initial_learning_rate = initial_learning_rate * multiplier
    boundaries = [int(steps_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]

    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
    if warmup:
        warmup_iter = float(steps_per_epoch * 5)
        _ratio = 1.0 / (multiplier * 4)
        warmup_ratio = tf.minimum(1.0, (1.0 - _ratio) * (tf.cast(global_step, tf.float32) / warmup_iter) ** 2 + _ratio)
        learning_rate *= warmup_ratio
    return learning_rate

def average_gradients(tower_grads):
    average_grads = []
    for grad in tower_grads:
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g in grad:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        average_grads.append(grad)
    return average_grads

