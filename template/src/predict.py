from __future__ import print_function
import argparse
import json
from importlib import import_module
import numpy as np


import chainer
from chainer import cuda
from chainer import serializers
from chainer.dataset import convert


import model
import util


def parse_args():
    parser = argparse.ArgumentParser(description='prediction')
    parser.add_argument('config_path',
                        help='Configuration file path')
    parser.add_argument('model',
                        help='Mofel file path')
    parser.add_argument('output', type=str, default=None,
                        help='Result output file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device ID (negative value indicates CPU)')
    return parser.parse_args()


def predict_dataset(net, iterator, converter=convert.concat_examples, device=None):
    scores = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for batch in iterator:
            x, t = converter(batch, device)
            y = model.predict(net, x)
            scores.append(cuda.to_cpu(y.data))
    return np.concatenate(scores, axis=0)


def main():
    args = parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    command_config = {'gpu': args.gpu}
    config.update(command_config)

    device_id = config['gpu']
    batch_size = config['batch_size']
    output_path = args.output

    network_params = config['network']
    for k, v in network_params.items():
        net = util.create_network(v)
        model_path = '{}.{}.model'.format(args.model, k)
        serializers.load_npz(model_path, net)
    if device_id >= 0:
        chainer.cuda.get_device_from_id(device_id).use()
        for net in nets.values():
            net.to_gpu()

    datasets = util.get_dataset(config.get('dataset', {}))
    test = datasets['test']
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)
    y = predict_dataset(net, test_iter, device=device_id)
    np.save(output_path, y)


if __name__ == '__main__':
    main()
