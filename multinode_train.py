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
    device = devices.get_devices(args.num_gpus)
    num_workers = len(configs['cluster_spec']['worker'])
    params = configs['params']
    params['steps_per_epoch'] = params['dataset']['images'] // (params['batchsize'] * device['count'] * num_workers)
    logging.info('\nargs=%s\nconfig=%s\ndevice=%s', args, configs, device)

    is_chief = args.task_index == 0
    cluster = tf.train.ClusterSpec(configs['cluster_spec'])
    server = tf.train.Server(cluster, job_name='worker', task_index=args.task_index)
    task_device_spec = tf.DeviceSpec(job="worker", task=args.task_index).to_string()
   
    thread = train.build_remote_feeder_thread(args.port, params['batchsize'])
    logging.info('build feeder thread')

    # build model graph
    models = []
    with tf.device(tf.train.replica_device_setter(worker_device=task_device_spec, cluster=cluster)):
        for device_index in range(device['count']):
            with tf.device(tf.DeviceSpec(device_type=device['name'], device_index=device_index)), \
                 tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                xs, labels = thread.tensors()

                model = resnet_model.ResNet(50, params['dataset']['classes'], xs, labels, is_training=True)
                model.build_graph()
                models.append(model)
    logging.info('build graph model')

    with tf.device(tf.train.replica_device_setter(worker_device=task_device_spec, cluster=cluster)):
        global_step = tf.train.get_or_create_global_step()

    with tf.device(tf.train.replica_device_setter(worker_device=task_device_spec, cluster=cluster)) and tf.device(tf.DeviceSpec(device_type=device['name'], device_index=0)):
        learning_rate = train.build_learning_rate(global_step, device['count']*num_workers, params)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        #optimizer = tf.train.SyncReplicasOptimizer(optimizer,
        #    replicas_to_aggregate=num_workers, total_num_replicas=num_workers)
    logging.info('build optimizer')

    with tf.device(tf.train.replica_device_setter(worker_device=task_device_spec, cluster=cluster)) and tf.device(tf.DeviceSpec(device_type=device['name'], device_index=0)):
        grads = train.average_gradients(zip(*[m.grads for m in models]))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), global_step=global_step)
        loss = tf.reduce_mean([m.loss for m in models], name='loss')
        accuracy = tf.reduce_mean([m.accuracy for m in models], name='accuracy')
        accuracy_top5 = tf.reduce_mean([m.accuracy_top5 for m in models], name='accuracy_top5')
    logging.info('build variables')

    with tf.device(tf.DeviceSpec(device_type='CPU', device_index=0)):
        checkpoint_saver = tf.train.CheckpointSaverHook(
            saver=tf.train.Saver(max_to_keep=100),
            checkpoint_dir=args.checkpoint_dir, save_steps=params['steps_per_epoch'])
        summary_saver = tf.train.SummarySaverHook(
            summary_op=tf.summary.merge_all(),
            output_dir=args.summary_dir, save_steps=params['steps_per_epoch']// 30)
        #sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        chief_hooks = [checkpoint_saver, summary_saver]
        hooks = [] #[sync_replicas_hook]
    logging.info('build hooks')

    fetches = {
        'ops': [train_op],
        'global_step': global_step, 'learning_rate': learning_rate,
        'loss': loss, 'accuracy': accuracy, 'accuracy_top5': accuracy_top5,
        'queue_size': thread.queue_size(),
    }

    # train loop
    session_config = tf.ConfigProto(
        intra_op_parallelism_threads=params['num_process_per_gpu']*device['count'],
        inter_op_parallelism_threads=params['num_process_per_gpu']*device['count']*2,
        allow_soft_placement=True, log_device_placement=False
    )
    with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            checkpoint_dir=args.checkpoint_dir, save_checkpoint_secs=None,
            save_summaries_steps=None, save_summaries_secs=None,
            hooks=hooks,
            chief_only_hooks=chief_hooks,
            config=session_config) as sess:
        logging.info('start session')
        thread.start(sess)
        logging.info('start feed data queue thread')

        for epoch in range(100):
            for step in range(params['steps_per_epoch']):
                start_time = time.time()
                results = sess.run(fetches)
                results.update({
                    'epoch': results['global_step'] // params['steps_per_epoch'], 'step': results['global_step'] % params['steps_per_epoch'],
                    'steps_per_epoch': params['steps_per_epoch'], 'batchsize': params['batchsize'], 'device_counts': device['count'],
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
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config json file')
    parser.add_argument('--task-index', type=int, default=0,
                        help='worker index')

    parser.add_argument('--name', type=str, required=True,
                        help='project name')
    parser.add_argument('-n', '--num-gpus', type=int, default=-1)
    parser.add_argument('--port',         type=int, required=True,
                        help='must be a set remote mode feeder')

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

    main(args)
