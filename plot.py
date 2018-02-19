import os
import sys
import logging
import re

import ujson as json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def parse(filename, elapsed_unit=60*60):
    value_list = []
    with open(filename, 'r') as f:
        accumulated_time = 0
        for line in f:
            if 'INFO' not in line or 'epoch' not in line:
                continue
            if 'rank' in line and 'rank:00' not in line:
                continue
            line = re.sub(r'\([^)]*\)|\[[^)]*\]|{|}', r'', line)
            items = [item.split(':') for item in line.split(' ') if item.strip()]

            values = {}
            for item in items:
                key = item[0]
                value = item[-1].replace(',', '').replace('sec', '')
                if '/' in value:
                    value = float(value.split('/')[0]) / float(value.split('/')[1])
                values[key] = float(value)
            values['elapsed'] = values['elapsed'] / elapsed_unit
            accumulated_time += values['elapsed']
            values['accumulated'] = accumulated_time
            value_list.append(values)
    return value_list

def sample(data, num_samples=100):
    step = len(data) // num_samples
    return data[::step]

def average(data, targets, steps=100):
    samples = []
    for i in range(len(data)//steps):
        sample = data[i*steps:i*steps+steps]

        result = {'step':sample[0]['step'], 'epoch':sample[0]['epoch'], 'accumulated':sample[0]['accumulated']}
        for t in targets:
            result[t] = np.average([d[t] for d in sample])
        samples.append(result)
    return samples

def main(args):
    matplotlib.rc('font', family="AppleGothic")

    logging.info(args)
    datas = [parse(filename) for filename in args.files]
    targets = ['loss', 'learning-rate', 'accuracy', 'top5']

    for xaxis in ['epoch', 'elapsed']:
        fig = plt.figure(figsize=(12, 8))
        for i, target in enumerate(targets):
            ax = fig.add_subplot(2, 2, i+1)
            ax.set_xlabel(xaxis.replace('elapsed', 'elapsed (hours)'), fontsize=12)
            ax.set_ylabel(target, fontsize=12)

            for filename, data in zip(args.files, datas):
                filename = filename.split('/')[-1]
                data = sample(data, args.samples) if args.sample else average(data, targets, args.samples)
                if xaxis == 'epoch':
                    ax.plot([d['step']+d['epoch'] for d in data], [d[target] for d in data], label=filename)
                if xaxis == 'elapsed':
                    ax.plot([d['accumulated'] for d in data], [d[target] for d in data], label=filename)
                if target in ['accuracy', 'top5']:
                    ax.set_ylim(0.0, 1.0)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.output + '-'+xaxis+'.png')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('-o', '--output',  type=str, default='result')
    parser.add_argument('-n', '--samples', type=int, default=100)
    parser.add_argument('--sample',        action='store_true')
    parser.add_argument('--log-filename',  type=str, default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main(args)

