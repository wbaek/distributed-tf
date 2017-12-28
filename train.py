import sys
import copy
import logging

import numpy as np
import cv2
import tensorflow as tf
import tensorpack as tp
import tensorpack.dataflow as df
import dataflow
from tensorflow.python.client import device_lib

from utils.imagenet import fbresnet_augmentor
from utils.tensorflow import average_gradients
from networks import resnet_model

def main(args):
    # data feeder
    augmentors = fbresnet_augmentor(isTrain=(args.mode is 'train'))
    if args.dataset == 'imagenet':
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
    #ds = df.PrefetchDataZMQ(ds, nr_proc=args.process)
    ds = df.BatchData(ds, args.batchsize, remainder=not (args.mode is 'train'))
    ds.reset_state()


    # networks
    devices = device_lib.list_local_devices()
    num_gpus = len([d for d in devices if '/gpu' in d.name])

    device_name = 'gpu' if num_gpus > 0 else 'cpu'
    device_counts = num_gpus if num_gpus > 0 else 1

    hps = resnet_model.HParams(batch_size=args.batchsize,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

    inputs = []
    models = []
    for device_idx in range(device_counts):
        with tf.device('/%s:%d'%(device_name, device_idx)),\
             tf.name_scope('TASK%d_TOWER%d'%(0, device_idx)),\
             tf.variable_scope(tf.get_variable_scope(), reuse=not (device_idx is 0)) as scope:
            input = tp.input_source.QueueInput(ds)
            input.setup([
                tp.graph_builder.InputDesc(tf.float32, (args.batchsize, 224, 224, 3), 'image'),
                tp.graph_builder.InputDesc(tf.int64, (args.batchsize, ), 'label')
            ])
            inputs.append( input )

            xs, labels = input._get_input_tensors()
            model = resnet_model.ResNet(hps, xs, labels, args.mode)
            model.build_graph()
            models.append( model )

            
    gradients_avg = average_gradients([zip(m.gradients, m.variables) for m in models])
    op = models[0].optimizer.apply_gradients(gradients_avg, global_step=models[0].global_step)
    cost = tf.reduce_mean([m.cost for m in models])
    accuracy = tf.reduce_mean([m.accuracy for m in models])

    #with tf.train.MonitoredTrainingSession(checkpoint_dir=args.checkpoint_dir) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for input in inputs:
            input.reset_state()
            input.thread.start()

        #while not sess.should_stop()
        for epoch in range(100):
            for step in range(ds.size() // (args.batchsize * device_counts)):
                _, c, a = sess.run([op, cost, accuracy])
                print('epoch:%03d step:%09d loss:%.6f accuracy:%.6f'%(epoch, step, c, a))


    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='mnist|cifar10|imagenet')
    parser.add_argument('--batchsize',    type=int, default=32)
    parser.add_argument('--service-code', type=str, default='',
                        help='licence key')
    parser.add_argument('-t', '--threads', type=int, default=16)
    parser.add_argument('-p', '--process', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train',
                        help='train or valid')
    parser.add_argument('--checkpoint-dir', type=dir, default='./checkpoints/')
    parser.add_argument('--log-filename',   type=str, default='')
    args = parser.parse_args()
    logging.getLogger("requests").setLevel(logging.WARNING)

    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main(args)
