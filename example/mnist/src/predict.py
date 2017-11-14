from __future__ import print_function
import argparse
import json
import numpy as np


import chainer
from chainer import serializers


import dataset
import prediction
from net.mlp import MLP


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST prediction example')
    parser.add_argument('config_path',
                        help='Configuration file path')
    parser.add_argument('model',
                        help='Mofel file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device ID (negative value indicates CPU)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Result output file path')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    command_config = {'gpu': args.gpu}
    config.update(command_config)

    device_id = config['gpu']
    batch_size = config['batch_size']
    output_path = args.output

    net = MLP(28 * 28, config['layers'], 10)
    if device_id >= 0:
        chainer.cuda.get_device_from_id(device_id).use()
        net.to_gpu()
    serializers.load_npz(args.model, net)

    _train, _valid, test = dataset.get_dataset()
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)
    y = prediction.predict_dataset(net, test_iter, device=device_id)
    np.save(output_path, y)


if __name__ == '__main__':
    main()
