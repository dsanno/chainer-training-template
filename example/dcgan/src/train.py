from __future__ import print_function
import argparse
from importlib import import_module
import json


import chainer
from chainer.training import Trainer
from chainer.training import extensions
from chainer.training import triggers


import dataset
import model
from training import TrainingStep
from training.extensions import generate_image
import util

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
    parser.add_argument('--app-config', type=str, default='config/app.json',
                        help='Application config file path')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    with open(args.app_config) as f:
        app_config = json.load(f)
    app_train_config = app_config.get('train', {})
    command_config = {'gpu': args.gpu}
    if args.output_dir is not None:
        command_config['output_dir'] = args.output_dir
    config.update(command_config)

    device_id = config['gpu']
    batch_size = config['batch_size']

    network_params = config['network']
    nets = {k: util.create_network(v) for k, v in network_params.items()}
    optimizers = {k: util.create_optimizer(v['optimizer'], nets[k])
            for k, v in network_params.items()}
    if len(optimizers) == 1:
        key, target_optimizer = list(optimizers.items())[0]
        target = nets[key]
    else:
        target = nets
        target_optimizer = optimizers

    if device_id >= 0:
        chainer.cuda.get_device_from_id(device_id).use()
        for net in nets.values():
            net.to_gpu()

    datasets = dataset.get_dataset()
    iterators = {}
    if isinstance(datasets, dict):
        for name, data in datasets.items():
            if name == 'train':
                train_iterator = chainer.iterators.SerialIterator(data, batch_size)
            else:
                iterators[name] = chainer.iterators.SerialIterator(data, batch_size,
                                                            repeat=False, shuffle=False)
    else:
        train_iterator = chainer.iterators.SerialIterator(datasets, batch_size)
    updater = TrainingStep(train_iterator, target_optimizer, model.calculate_metrics,
                           device=device_id)
    trainer = Trainer(updater, (config['epoch'], 'epoch'), out=config['output_dir'])
    if hasattr(model, 'make_eval_func'):
        for name, iterator in iterators.items():
            evaluator = extensions.Evaluator(iterator, target,
                                             eval_func=model.make_eval_func(target),
                                             device=device_id)
            trainer.extend(evaluator, name=name)

    dump_graph_node = app_train_config.get('dump_graph', None)
    if dump_graph_node is not None:
        trainer.extend(extensions.dump_graph(dump_graph_node))

    trainer.extend(extensions.snapshot(filename='snapshot.state'),
                   trigger=(1, 'epoch'))
    for k, net in nets.items():
        file_name = 'latest.{}.model'.format(k)
        trainer.extend(extensions.snapshot_object(net, filename=file_name),
                       trigger=(1, 'epoch'))
    max_value_trigger_key = app_train_config.get('max_value_trigger', None)
    min_value_trigger_key = app_train_config.get('min_value_trigger', None)
    if max_value_trigger_key is not None:
        trigger = triggers.MaxValueTrigger(max_value_trigger_key)
        for key, net in nets.items():
            file_name = 'best.{}.model'.format(key)
            trainer.extend(extensions.snapshot_object(net, filename=file_name),
                           trigger=trigger)
    elif min_value_trigger_key is not None:
        trigger = triggers.MinValueTrigger(min_value_trigger_key)
        for key, net in nets.items():
            file_name = 'best.{}.model'.format(key)
            trainer.extend(extensions.snapshot_object(net, file_name),
                           trigger=trigger)
    trainer.extend(extensions.LogReport())
    if len(optimizers) == 1:
        for name, opt in optimizers.items():
            if not hasattr(opt, 'lr'):
                continue
            trainer.extend(extensions.observe_lr(name))
    else:
        for name, opt in optimizers.items():
            if not hasattr(opt, 'lr'):
                continue
            key = '{}/lr'.format(name)
            trainer.extend(extensions.observe_lr(name, key))

    if extensions.PlotReport.available():
        plot_targets = app_train_config.get('plot_report', {})
        for name, targets in plot_targets.items():
            file_name = '{}.png'.format(name)
            trainer.extend(extensions.PlotReport(targets, 'epoch',
                file_name=file_name))

    if not args.silent:
        print_targets = app_train_config.get('print_report', [])
        if print_targets is not None and print_targets != []:
            trainer.extend(extensions.PrintReport(print_targets))
        trainer.extend(extensions.ProgressBar())

    trainer.extend(generate_image(nets['gen'], 10, 10,
                   config['output_image_dir']), trigger=(1, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
