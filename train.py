import os
import shutil
import sys
import time
import logging

import cv2
import tensorflow as tf
import tensorpack.dataflow as df
import dataflow
import dataflow.tensorflow
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline

from utils.imagenet import fbresnet_augmentor
from utils.utils import average_gradients
from networks import resnet_model


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(args.num_gpus)])
    devices = device_lib.list_local_devices()
    num_gpus = len([d for d in devices if 'GPU' in d.device_type])

    device_name = 'GPU' if num_gpus > 0 else 'CPU'
    device_counts = min(num_gpus if num_gpus > 0 else 1, args.num_gpus)
    logging.info({'devices': devices, 'device_name': device_name, 'device_counts': device_counts})

    args.num_gpus = device_counts
    args.process *= device_counts
    logging.info(args)

    dataset_meta = {
        'num_classes':{
            'imagenet': 1000,
            'cifar10': 10,
            'mnist': 10,
        },
        'num_images':{
            'imagenet': 1281167,
            'cifar10': 50000,
            'mnist': 60000,
        }
    }
    ds = df.RemoteDataZMQ('tcp://0.0.0.0:' + str(args.port))
    ds = df.BatchData(ds, args.batchsize, remainder=False)
    ds = df.PrefetchDataZMQ(ds, nr_proc=1)
    ds.reset_state()
    num_classes = dataset_meta['num_classes'][args.dataset]
    num_images = dataset_meta['num_images'][args.dataset]
    steps_per_epoch = num_images // (args.batchsize * device_counts)
    logging.info('build remote feed data (tcp://0.0.0.0:%s)'%str(args.port))

    # feed data queue input
    with tf.device(tf.DeviceSpec(device_type='CPU', device_index=0)):
        _placeholders = [
            tf.placeholder(tf.float32, (args.batchsize, 224, 224, 3)),
            tf.placeholder(tf.int64, (args.batchsize,))
        ]
        thread = dataflow.tensorflow.QueueInput(
            ds, _placeholders, repeat_infinite=False, queue_size=50)
    logging.info('build feed data queue thread')

    # build model graph
    models = []
    for device_idx in range(device_counts):
        with tf.device(tf.DeviceSpec(device_type=device_name, device_index=device_idx)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            xs, labels = thread.tensors()

            model = resnet_model.ResNet(50, num_classes, xs, labels, is_training=True)
            model.build_graph()
            models.append(model)
    logging.info('build graph model')

    with tf.device(tf.DeviceSpec(device_type=device_name, device_index=0)):
        loss = tf.reduce_mean([m.loss for m in models], name='loss')
        accuracy = tf.reduce_mean([m.accuracy for m in models], name='accuracy')
        accuracy_top5 = tf.reduce_mean([m.accuracy_top5 for m in models], name='accuracy_top5')
        global_step = tf.train.get_or_create_global_step()

        learning_rate_multiplier = (args.batchsize * device_counts) / 256.0
        initial_learning_rate = args.learning_rate * learning_rate_multiplier
        boundaries = [int(steps_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
        if args.warmup:
            warmup_iter = float(steps_per_epoch * 5)
            _ratio = 1.0 / (learning_rate_multiplier * 4)
            warmup_ratio = tf.minimum(1.0, (1.0 - _ratio) * (tf.cast(global_step, tf.float32) / warmup_iter) ** 2 + _ratio)
            learning_rate *= warmup_ratio

        trainable_variables = tf.trainable_variables()
        tower_grads = zip(*[m.grads for m in models])
        grads = average_gradients(tower_grads)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step)

    logging.info('build optimizer')

    with tf.device(tf.DeviceSpec(device_type='CPU', device_index=0)):
        checkpoint_saver = tf.train.CheckpointSaverHook(
            saver = tf.train.Saver(max_to_keep=100),
            checkpoint_dir = args.checkpoint_dir, save_steps=steps_per_epoch)
        summary_saver = tf.train.SummarySaverHook(
            summary_op = tf.summary.merge_all(),
            output_dir = args.summary_dir, save_steps=steps_per_epoch // 30)
        hooks = [checkpoint_saver, summary_saver]
    logging.info('build hooks')

    fetches = {
        'ops': [train_op],
        'global_step': global_step,
        'loss': loss,
        'accuracy': accuracy,
        'accuracy_top5': accuracy_top5,
        'learning_rate': learning_rate,
        'queue_size': thread.queue_size(),
    }

    if args.profile:
        folder = './profile/%s/'%args.name
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # train loop
    config = tf.ConfigProto(
        intra_op_parallelism_threads=args.process, inter_op_parallelism_threads=args.process*2,
        allow_soft_placement=True, log_device_placement=args.profile
    )

    if args.profile: 
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
    else:
        sess = tf.train.SingularMonitoredSession(
         config=config, hooks=hooks, checkpoint_dir=args.checkpoint_dir)

    with sess:
        thread.start(sess)
        logging.info('start feed data queue thread')

        for epoch in range(100):
            for step in range(steps_per_epoch):
                start_time = time.time()
                if args.profile:
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    results = sess.run(fetches, options=options, run_metadata=run_metadata)

                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('./profile/%s/timeline_%04d.json'%(args.name, results['global_step']), 'w') as f:
                        f.write(chrome_trace)
                else:
                    results = sess.run(fetches)
                results.update({
                    'epoch': results['global_step']//steps_per_epoch,
                    'step': results['global_step']%steps_per_epoch,
                    'steps_per_epoch': steps_per_epoch,
                    'batchsize': args.batchsize,
                    'device_counts': device_counts,
                    'elapsed': time.time() - start_time
                })
                results['images_per_sec'] = (results['batchsize'] * results['device_counts']) / results['elapsed']
                logging.info(
                    'epoch:{epoch:03d} step:{step:04d}/{steps_per_epoch:04d} '
                    'learning-rate:{learning_rate:.3f} '
                    'loss:{loss:.4f} accuracy:{{top1:{accuracy:.4f}, top5:{accuracy_top5:.4f}}} '
                    'elapsed:{elapsed:.1f}sec '
                    '({images_per_sec:.3f}images/sec queue:{queue_size})'.format_map(results))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--name', type=str, required=True,
                        help='project name')

    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='mnist|cifar10|imagenet')
    parser.add_argument('-n', '--num-gpus', type=int, default=8)
    parser.add_argument('--process',        type=int, default=2)

    parser.add_argument('--batchsize',    type=int, default=128)
    parser.add_argument('--port',         type=int, required=True,
                        help='must be a set remote mode feeder')

    parser.add_argument('--warmup',        action='store_true')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='learning rate based on batchsize=256 (default=0.01)')

    parser.add_argument('--profile', action='store_true')

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

