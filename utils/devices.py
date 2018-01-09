from tensorflow.python.client import device_lib

import logging
logger = logging.getLogger(__name__)


def get_devices(max_gpus=-1):
    if max_gpus >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(max_gpus)])
    devices = device_lib.list_local_devices()
    num_gpus = len([d for d in devices if 'GPU' in d.device_type])

    device_name = 'GPU' if num_gpus > 0 else 'CPU'
    device_counts = num_gpus if num_gpus > 0 else 1
    return {'name':device_name, 'count':device_counts}

