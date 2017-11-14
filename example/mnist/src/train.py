from __future__ import print_function
import argparse
import json


import chainer
from chainer.training import Trainer
from chainer.training import extensions
from chainer.training import triggers


import dataset
from net.mlp import MLP
import training
from training import TrainingStep


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST example')
    parser.add_argument('config_path',
                        help='Configuration file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device ID (negative value indicates CPU)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Result output directory')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume file path')
    parser.add_argument('--silent', action='store_true',
                        help='Do not print training progress if set')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    command_config = {'gpu': args.gpu}
    if args.output_dir is not None:
        command_config['output_dir'] = args.output_dir
    config.update(command_config)

    device_id = config['gpu']
    batch_size = config['batch_size']

    net = MLP(28 * 28, config['layers'], 10)
    if device_id >= 0:
        chainer.cuda.get_device_from_id(device_id).use()
        net.to_gpu()

    optimizer = chainer.optimizers.Adam(config['learning_rate'])
    optimizer.setup(net)

    train, valid, test = dataset.get_dataset()

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    valid_iter = chainer.iterators.SerialIterator(valid, batch_size,
                                                 repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

    updater = TrainingStep(train_iter, optimizer, training.calculate_metrics,
                           device=device_id)
    trainer = Trainer(updater, (config['epoch'], 'epoch'),
                               out=config['output_dir'])

    evaluator = extensions.Evaluator(valid_iter, net,
                                     eval_func=training.make_eval_func(net),
                                     device=device_id)
    trainer.extend(evaluator, name='validation')
    evaluator = extensions.Evaluator(test_iter, net,
                                     eval_func=training.make_eval_func(net),
                                     device=device_id)
    trainer.extend(evaluator, name='test')

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(filename='snapshot.state'),
                   trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(net, filename='latest.model'),
                   trigger=(1, 'epoch'))
    trigger = triggers.MaxValueTrigger('validation/main/accuracy')
    trainer.extend(extensions.snapshot_object(net, filename='best.model'),
                   trigger=trigger)

    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'test/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'test/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    if not args.silent:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'main/accuracy',
             'validation/main/accuracy', 'test/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
