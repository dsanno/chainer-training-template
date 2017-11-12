from __future__ import print_function
import argparse
import json
import numpy as np
import os


import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import triggers


import dataset
import method
from net.dcgan import Generator, Discriminator
from training.training_step import TrainingStep
from training.generate_image import generate_image


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
    latent_size = config['latent_size']
    output_dir = config['output_dir']
    out_image_dir = os.path.join(output_dir, 'image')
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)

    gen = Generator(latent_size)
    dis = Discriminator()
    if device_id >= 0:
        chainer.cuda.get_device_from_id(device_id).use()
        gen.to_gpu()
        dis.to_gpu()

    gen_optimizer = chainer.optimizers.Adam(config['learning_rate'])
    gen_optimizer.setup(gen)
    dis_optimizer = chainer.optimizers.Adam(config['learning_rate'])
    dis_optimizer.setup(dis)
    optimizers = {
        'gen': gen_optimizer,
        'dis': dis_optimizer,
    }

    train, _test = dataset.get_dataset()

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    updater = TrainingStep(train_iter, optimizers, method.calculate_metrics,
                           device=device_id)
    trainer = training.Trainer(updater, (config['epoch'], 'epoch'),
                               out=output_dir)

    trainer.extend(extensions.snapshot(filename='snapshot.state'),
                   trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(gen, filename='gen.model'),
                   trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(gen, filename='dis.model'),
                   trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport())

    trainer.extend(generate_image(gen, 10, 10, out_image_dir),
                   trigger=(1, 'epoch'))

    if not args.silent:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'gen/loss', 'dis/loss', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
