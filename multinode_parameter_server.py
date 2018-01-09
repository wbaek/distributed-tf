import os
import sys
import logging

import ujson as json
import tensorflow as tf


def main(args):
    with open(args.config, 'r') as f:
        configs = json.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    cluster = tf.train.ClusterSpec(configs['cluster_spec'])
    server = tf.train.Server(cluster, job_name='ps', task_index=args.task_index)
    server.join()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Distributed Training Parameter Server')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config json file')
    parser.add_argument('--task-index', type=int, default=0,
                        help='parameter server index')
    parser.add_argument('--log-filename',   type=str, default='')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log_filename)

    main(args)
