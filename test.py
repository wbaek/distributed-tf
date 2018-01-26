import os
import sys
import time
import logging

import ujson as json
import numpy as np
import tensorflow as tf
import tensorpack.dataflow as df

from utils import utils, devices, train
from networks import resnet_model

from remote_feeder import get_datastream

def main(args):
    with open(args.config, 'r') as f:
        configs = json.load(f)
    device = devices.get_devices(gpu_ids=args.gpus)
    params = configs['params']
    params['steps_per_epoch'] = params['test']['images'] // (params['test']['batchsize'] * device['count'])
    logging.info('\nargs=%s\nconfig=%s\ndevice=%s', args, configs, device)

    with tf.device(devices.get_device_spec(device, _next=True)):
        ds, num_classes = get_datastream(params['dataset']['name'], params['test']['mode'],
            params['test']['batchsize'],
            args.service_code, params['test']['processes'], params['test']['threads'],
            shuffle=False, remainder=True)
        ds = df.RepeatedData(ds, 2)
        ds.reset_state()
        thread = train.build_ds_thread(ds, params['test']['batchsize'], (224, 224, 3), queue_size=device['count']*1)
    logging.info('build feeder thread')

    # build model graph
    models = []
    for device_index in range(device['count']):
        device_spec = tf.DeviceSpec(device_type=device['name'], device_index=device_index)
        with tf.device(tf.train.replica_device_setter(worker_device=device_spec.to_string(), ps_device='/cpu:0', ps_tasks=1)), \
             tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            xs, labels = thread.tensors()

            model = resnet_model.ResNet(50, params['dataset']['classes'], xs, labels, is_training=False)
            model.build_graph()
            models.append(model)
    logging.info('build graph model') 

    with tf.device(devices.get_device_spec(device, _next=True)):
        loss = tf.reduce_mean([m.loss for m in models], name='loss')
        accuracy = tf.reduce_mean([m.accuracy for m in models], name='accuracy')
        accuracy_top5 = tf.reduce_mean([m.accuracy_top5 for m in models], name='accuracy_top5')
    logging.info('build variables')

    # session hooks
    saver = tf.train.Saver()
    fetches = {
        'loss': loss,
        'accuracy': accuracy,
        'accuracy_top5': accuracy_top5,
    }
    results = {}

    # train loop
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint)

        thread.start(sess)
        logging.info('start feed data queue thread')

        for step in range(params['steps_per_epoch']):
            start_time = time.time()
            results_batch = sess.run(fetches)
            results_batch.update({
                'step': step,
                'steps_per_epoch': params['steps_per_epoch'],
                'batchsize': params['test']['batchsize'],
                'elapsed': time.time() - start_time
            })
            results_batch['images_per_sec'] = results_batch['batchsize'] / results_batch['elapsed']
            logging.info(
                'step:{step:04d}/{steps_per_epoch:04d} '
                'loss:{loss:.4f} accuracy:{{top1:{accuracy:.4f}, top5:{accuracy_top5:.4f}}} '
                'elapsed:{elapsed:.1f}sec '
                '{images_per_sec:.3f}images/sec'.format_map(results_batch))
            for key in fetches.keys():
                if key not in results:
                    results[key] = []
                results[key].append( results_batch[key] )

    for key in results.keys():
        logging.info('{} = {:.5f}'.format(key, np.mean(results[key])))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config json file')
    parser.add_argument('-s', '--checkpoint', type=str, required=True)
    parser.add_argument('-g', '--gpus', nargs='*',
                        help='set gpu index (--gpus 0, 1)')

    parser.add_argument('--service-code', type=str, required=True,
                        help='licence key')

    parser.add_argument('--log-filename',   type=str, default='')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log_filename)
    logging.getLogger("requests").setLevel(logging.WARNING)

    main(args)
