import sys
import copy
import logging

import numpy as np
import cv2
import tensorflow as tf
import tensorpack as tp
import tensorpack.dataflow as df
from tensorpack.input_source.input_source import EnqueueThread
import dataflow
from tensorflow.python.client import device_lib

from utils.imagenet import fbresnet_augmentor
from utils.tensorflow import average_gradients
from networks import resnet_model

def get_datastream(args):
    # data feeder
    augmentors = fbresnet_augmentor(isTrain=(args.mode == 'train'))
    if args.dataset == 'imagenet':
        if len(args.service_code) < 0:
            raise ValueError('image is must be a set service-code')
        ds = dataflow.dataset.ILSVRC12(args.service_code, args.mode, shuffle=True).parallel(num_threads=args.threads)
        num_classes = 1000
    elif args.dataset == 'mnist':
        ds = df.dataset.Mnist(args.mode, shuffle=True)
        augmentors = [
            df.imgaug.MapImage(lambda x: x.reshape(28, 28, 1)),
            df.imgaug.ColorSpace(cv2.COLOR_GRAY2BGR),
            df.imgaug.Resize((256, 256))
        ] + augmentors
        num_classes = 10
    elif args.dataset == 'cifar10':
        ds = df.dataset.Cifar10(args.mode, shuffle=True)
        augmentors = [
            df.imgaug.Resize((256, 256))
        ] + augmentors
        num_classes = 10
    else:
        raise ValueError('%s is not support dataset'%args.dataset)
 
    ds = df.AugmentImageComponent(ds, augmentors, copy=False)
    ds = df.PrefetchDataZMQ(ds, nr_proc=args.process)
    ds = df.BatchData(ds, args.batchsize, remainder=not (args.mode == 'train'))
    ds.reset_state()

    return ds, num_classes

def main(args):
    devices = device_lib.list_local_devices()
    num_gpus = len([d for d in devices if 'GPU' in d.device_type])

    device_name = 'GPU' if num_gpus > 0 else 'CPU'
    device_counts = num_gpus if num_gpus > 0 else 1
    logging.info({'devices':devices, 'device_name':device_name, 'device_counts':device_counts})

    # feed data queue input
    with tf.device(tf.DeviceSpec(device_type=device_name, device_index=0)):
        placeholders = [
            tf.placeholder(tf.float32, (args.batchsize, 224, 224, 3)),
            tf.placeholder(tf.int64, (args.batchsize,))
        ]
        ds, num_classes = get_datastream(args)
        thread = dataflow.tensorflow.QueueInput(ds, placeholders, repeat_infinite=True)
    dp_splited = [tf.split(t, num_gpus) for t in thread.tensors()]
    logging.info('build feed data queue')

    hps = resnet_model.HParams(batch_size=args.batchsize//device_counts,
                               num_classes=num_classes,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1)
    models = []
    for device_idx in range(device_counts):
        with tf.device(tf.DeviceSpec(device_type=device_name, device_index=device_idx)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
            xs, labels = [dp[device_idx] for dp in dp_splited]

            model = resnet_model.ResNet(hps, xs, labels, args.mode)
            model.build_graph()
            models.append( model )
    logging.info('build graph model')

    with tf.device(tf.DeviceSpec(device_type=device_name, device_index=0)):
        cost = tf.reduce_mean([m.cost for m in models])
        accuracy = tf.reduce_mean([m.accuracy for m in models])
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
                                                   decay_steps=50000, decay_rate=0.8, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(cost, global_step, colocate_gradients_with_ops=True)
    logging.info('build optimizer')

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        thread.start()

        steps_per_epoch = ds.size() // (device_counts)
        for epoch in range(100):
            for step in range(steps_per_epoch):
                _, c, a = sess.run([train_op, cost, accuracy])
                logging.info('epoch:%03d step:%06d/%06d loss:%.6f accuracy:%.6f',
                    epoch, step, steps_per_epoch, c, a)

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='mnist|cifar10|imagenet')
    parser.add_argument('--batchsize',    type=int, default=32)
    parser.add_argument('--service-code', type=str, default='',
                        help='licence key')
    parser.add_argument('-t', '--threads', type=int, default=4)
    parser.add_argument('-p', '--process', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train',
                        help='train or valid')

    parser.add_argument('-l', '--learning-rate', type=float, default=0.01)

    parser.add_argument('--checkpoint-dir', type=dir, default='./checkpoints/')
    parser.add_argument('--log-filename',   type=str, default='')
    args = parser.parse_args()
    logging.getLogger("requests").setLevel(logging.INFO)

    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main(args)
