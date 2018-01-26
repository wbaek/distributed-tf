import sys
import logging

import cv2
import tensorpack.dataflow as df
import dataflow

from utils.imagenet import fbresnet_augmentor


def get_datastream(dataset, mode, batchsize=0, service_code=None, processes=1, threads=1, shuffle=True, remainder=False):
    # data feeder
    augmentors = fbresnet_augmentor(isTrain=(mode == 'train'))
    if dataset == 'imagenet':
        if len(service_code) < 0:
            raise ValueError('image is must be a set service-code')
        ds = dataflow.dataset.ILSVRC12(service_code, mode, shuffle=shuffle).parallel(num_threads=threads)
        num_classes = 1000
    elif dataset == 'mnist':
        ds = df.dataset.Mnist(mode, shuffle=shuffle)
        augmentors = [
            df.imgaug.MapImage(lambda x: x.reshape(28, 28, 1)),
            df.imgaug.ColorSpace(cv2.COLOR_GRAY2BGR),
            df.imgaug.Resize((256, 256))
        ] + augmentors
        num_classes = 10
    elif dataset == 'cifar10':
        ds = df.dataset.Cifar10(mode, shuffle=shuffle)
        augmentors = [
            df.imgaug.Resize((256, 256))
        ] + augmentors
        num_classes = 10
    else:
        raise ValueError('%s is not support dataset' % dataset)
 
    ds = df.AugmentImageComponent(ds, augmentors, copy=False)
    if batchsize > 0:
        ds = df.BatchData(ds, batchsize, remainder=remainder)
    ds = df.PrefetchDataZMQ(ds, nr_proc=processes)
    return ds, num_classes


def main(args):
    logging.info(args)

    ds, num_classes = get_datastream(
        args.dataset, args.mode, args.batchsize,
        args.service_code, args.process, args.threads)
    # ds = df.RepeatedData(ds, -1)
    logging.info('build feed data')

    while True:
        try:
            ds.reset_state()
            df.send_dataflow_zmq(ds, args.target)
        except Exception as e:
            logging.warning('%s exception: %s', str(e))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Imagenet Dataset Feeder')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='mnist|cifar10|imagenet')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or valid or test')
    parser.add_argument('--batchsize', type=int, default=128)

    parser.add_argument('--service-code', type=str, default='',
                        help='licence key')
    parser.add_argument('-p', '--process', type=int, default=6)
    parser.add_argument('-t', '--threads', type=int, default=8)

    parser.add_argument('--target', type=str, required=True,
                        help='target host and port (ex:tcp://localhost:2222)')

    parser.add_argument('--log-filename',   type=str, default='')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log_filename)
    logging.getLogger("requests").setLevel(logging.WARNING)

    main(args)
