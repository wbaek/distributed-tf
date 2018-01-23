import os

import tensorflow as tf
from tensorflow.python.client import device_lib

import logging
logger = logging.getLogger(__name__)


def get_devices(gpu_ids=[], max_gpus=-1):
    if gpu_ids is None or (len(gpu_ids) == 0 and max_gpus > 0):
        gpu_ids = list(range(max_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    devices = device_lib.list_local_devices()
    num_gpus = len([d for d in devices if 'GPU' in d.device_type])

    device_name = 'GPU' if num_gpus > 0 else 'CPU'
    device_counts = num_gpus if num_gpus > 0 else 1
    return {'name':device_name, 'count':device_counts}

current_index = 0
def get_device_spec(device, _next=False):
    global current_index
    if _next:
        current_index = current_index + 1
        current_index = current_index % device['count']
    return tf.DeviceSpec(device_type=device['name'], device_index=current_index)

