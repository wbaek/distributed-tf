import os
import sys
import time
import logging

import cv2
import tensorflow as tf
import tensorpack.dataflow as df
import dataflow
from tensorflow.python.client import device_lib

from utils.imagenet import fbresnet_augmentor
from networks import resnet_model


def get_datastream(dataset, mode, batchsize, service_code=None, processes=1, threads=1):
    # data feeder
    augmentors = fbresnet_augmentor(isTrain=(mode == 'train'))
    if dataset == 'imagenet':
        if len(service_code) < 0:
            raise ValueError('image is must be a set service-code')
        ds = dataflow.dataset.ILSVRC12(service_code, mode, shuffle=True).parallel(num_threads=threads)
        num_classes = 1000
    elif dataset == 'mnist':
        ds = df.dataset.Mnist(mode, shuffle=True)
        augmentors = [
            df.imgaug.MapImage(lambda x: x.reshape(28, 28, 1)),
            df.imgaug.ColorSpace(cv2.COLOR_GRAY2BGR),
            df.imgaug.Resize((256, 256))
        ] + augmentors
        num_classes = 10
    elif dataset == 'cifar10':
        ds = df.dataset.Cifar10(mode, shuffle=True)
        augmentors = [
            df.imgaug.Resize((256, 256))
        ] + augmentors
        num_classes = 10
    else:
        raise ValueError('%s is not support dataset' % dataset)
 
    ds = df.AugmentImageComponent(ds, augmentors, copy=False)
    ds = df.PrefetchDataZMQ(ds, nr_proc=processes)
    ds = df.BatchData(ds, batchsize, remainder=not (mode == 'train'))
    ds.reset_state()

    return ds, num_classes


def main(args):
    logging.info(args)

    devices = device_lib.list_local_devices()
    num_gpus = len([d for d in devices if 'GPU' in d.device_type])

    device_name = 'GPU' if num_gpus > 0 else 'CPU'
    device_counts = num_gpus if num_gpus > 0 else 1
    logging.info({'devices': devices, 'device_name': device_name, 'device_counts': device_counts})

    # feed data queue input
    with tf.device(tf.DeviceSpec(device_type=device_name, device_index=0)):
        placeholders = [
            tf.placeholder(tf.float32, (args.batchsize, 224, 224, 3)),
            tf.placeholder(tf.int64, (args.batchsize,))
        ]
        ds, num_classes = get_datastream(
            args.dataset, args.mode, args.batchsize,
            args.service_code, args.process, args.threads)
        thread = dataflow.tensorflow.QueueInput(
            ds, placeholders, repeat_infinite=True, queue_size=5)
    dp_splited = [tf.split(t, device_counts) for t in thread.tensors()]
    logging.info('build feed data queue thread')

    hps = resnet_model.HParams(batch_size=args.batchsize//device_counts,
                               num_classes=num_classes,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1)
    models = []
    for device_idx in range(device_counts):
        with tf.device(tf.DeviceSpec(device_type=device_name, device_index=device_idx)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            xs, labels = [dp[device_idx] for dp in dp_splited]

            model = resnet_model.ResNet(hps, xs, labels, args.mode)
            model.build_graph()
            models.append(model)
    logging.info('build graph model')

    with tf.device(tf.DeviceSpec(device_type=device_name, device_index=0)):
        cost = tf.reduce_mean([m.cost for m in models], name='cost')
        accuracy = tf.reduce_mean([m.accuracy for m in models], name='accuracy')
        accuracy_top5 = tf.reduce_mean([m.accuracy_top5 for m in models], name='accuracy_top5')
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            args.learning_rate, global_step,
            decay_steps=50000, decay_rate=0.8, staircase=True, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(cost, global_step, colocate_gradients_with_ops=True)
    logging.info('build optimizer')

    steps_per_epoch = ds.size() // device_counts
    checkpoint_saver = tf.train.CheckpointSaverHook(
        checkpoint_dir=args.checkpoint_dir, save_steps=steps_per_epoch)
    summary_saver = tf.train.SummarySaverHook(
        summary_op=tf.summary.merge_all(),
        output_dir=args.summary_dir, save_steps=steps_per_epoch//10)
    hooks = [checkpoint_saver, summary_saver]
    logging.info('build hooks')

    fetches = {
        'ops':[train_op],
        'global_step':global_step,
        'cost':cost,
        'accuracy':accuracy,
        'accuracy_top5':accuracy_top5,
        'learning_rate':learning_rate,
        'queue_size':thread.queue_size()
    }

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.train.SingularMonitoredSession(
         config=config, hooks=hooks, checkpoint_dir=args.checkpoint_dir) as sess:
        thread.start(sess)
        logging.info('start feed data queue thread')

        for epoch in range(100):
            for step in range(steps_per_epoch):
                start_time = time.time()
                results = sess.run(fetches)
                results.update({
                    'epoch':epoch,
                    'step':step,
                    'steps_per_epoch':steps_per_epoch,
                    'batchsize':args.batchsize,
                    'elapsed':time.time() - start_time
                })
                logging.info('epoch:{epoch:03d} step:{step:04d}/{steps_per_epoch:04d} loss:{cost:.4f} accuracy:{{top1:{accuracy:.4f}, top5:{accuracy_top5:.4f}}} elapsed:{elapsed:.1f}sec'.format_map(results))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--name', type=str, required=True,
                        help='project name')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='mnist|cifar10|imagenet')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or valid or test')
    parser.add_argument('--batchsize',    type=int, default=64)
    parser.add_argument('--service-code', type=str, default='',
                        help='licence key')
    parser.add_argument('-p', '--process', type=int, default=4)
    parser.add_argument('-t', '--threads', type=int, default=4)

    parser.add_argument('-l', '--learning-rate', type=float, default=0.01)

    currnet_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--checkpoint-dir', type=str, default=currnet_path+'/checkpoints/')
    parser.add_argument('--summary-dir', type=str, default=currnet_path+'/summaries/')
    parser.add_argument('--log-filename',   type=str, default='')
    args = parser.parse_args()
    args.checkpoint_dir += args.name + '/'
    args.summary_dir += args.name + '/'

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log_filename)
    logging.getLogger("requests").setLevel(logging.WARNING)

    main(args)
