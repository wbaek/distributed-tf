import os
import shutil
import sys
import time
import logging

import ujson as json
import tensorflow as tf

from utils import utils, train, devices
from networks import resnet_model


def main(args):
    with open(args.config, 'r') as f:
        configs = json.load(f)
    device = devices.get_devices(gpu_ids=args.gpus)
    params = configs['params']
    params['steps_per_epoch'] = params['dataset']['images'] // (params['batchsize'] * device['count'])
    logging.info('\nargs=%s\nconfig=%s\ndevice=%s', args, configs, device)

    thread = train.build_remote_feeder_thread(args.port, params['batchsize'], queue_size=device['count']*5)
    logging.info('build feeder thread')

    # build model graph
    models = []
    for device_index in range(device['count']):
        with tf.device(tf.DeviceSpec(device_type=device['name'], device_index=device_index)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            xs, labels = thread.tensors()

            model = resnet_model.ResNet(50, params['dataset']['classes'], xs, labels, is_training=True)
            model.build_graph()
            models.append(model)
    logging.info('build graph model')

    global_step = tf.train.get_or_create_global_step()
    learning_rate = train.build_learning_rate(global_step, device['count'], params)
    loss = tf.reduce_mean([m.loss for m in models], name='loss')
    accuracy = tf.reduce_mean([m.accuracy for m in models], name='accuracy')
    accuracy_top5 = tf.reduce_mean([m.accuracy_top5 for m in models], name='accuracy_top5')
    logging.info('build variables')

    grads = train.average_gradients(zip(*[m.grads for m in models]))
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), global_step=global_step)
    logging.info('build optimizer')

    checkpoint_saver = tf.train.CheckpointSaverHook(
        saver=tf.train.Saver(max_to_keep=100),
        checkpoint_dir=args.checkpoint_dir, save_steps=params['steps_per_epoch'])
    hooks = [checkpoint_saver]
    logging.info('build hooks')

    fetches = {
        'ops': train_op,
        'global_step': global_step, 'learning_rate': learning_rate,
        'loss': loss, 'accuracy': accuracy, 'accuracy_top5': accuracy_top5,
        'queue_size': thread.queue_size(),
    }

    config = tf.ConfigProto(
        intra_op_parallelism_threads=params['num_process_per_gpu']*device['count'],
        inter_op_parallelism_threads=params['num_process_per_gpu']*device['count']*2,
        allow_soft_placement=True, log_device_placement=args.profile,
        #gpu_options = tf.GPUOptions(allow_growth=False, force_gpu_compatible=True),
    )
    with tf.train.SingularMonitoredSession(config=config, hooks=hooks, checkpoint_dir=args.checkpoint_dir) as sess:
        thread.start(sess)
        logging.info('start feed data queue thread')

        for epoch in range(100):
            for step in range(params['steps_per_epoch']):
                start_time = time.time()
                results = utils.run_session_with_profile(sess, fetches, profile_dir='./profile/%s/'%args.name) if args.profile else sess.run(fetches)
                results.update({
                    'epoch': results['global_step'] // params['steps_per_epoch'], 'step': results['global_step'] % params['steps_per_epoch'],
                    'steps_per_epoch': params['steps_per_epoch'], 'batchsize': params['batchsize'], 'device_counts': device['count'],
                    'elapsed': time.time() - start_time
                })
                results['images_per_sec'] = (results['batchsize'] * results['device_counts']) / results['elapsed']
                logging.info(
                    'epoch:{epoch:03d} step:{step:04d}/{steps_per_epoch:04d} '
                    'learning-rate:{learning_rate:.5f} '
                    'loss:{loss:.4f} accuracy:{{top1:{accuracy:.4f}, top5:{accuracy_top5:.4f}}} '
                    'elapsed:{elapsed:.3f}sec '
                    '({images_per_sec:.1f}images/sec queue:{queue_size})'.format_map(results))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config json file')
    parser.add_argument('--name', type=str, required=True,
                        help='project name')
    parser.add_argument('-g', '--gpus', nargs='*',
                        help='set gpu index (--gpus 0, 1)')
    parser.add_argument('--port',         type=int, required=True,
                        help='must be a set remote mode feeder')

    currnet_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--checkpoint-dir', type=str, default=currnet_path+'/checkpoints/')
    parser.add_argument('--summary-dir', type=str, default=currnet_path+'/summaries/')
    parser.add_argument('--log-filename',   type=str, default='')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    args.checkpoint_dir += args.name + '/'
    args.summary_dir += args.name + '/'

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log_filename)

    main(args)
