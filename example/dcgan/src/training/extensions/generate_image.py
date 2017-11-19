import os
import numpy as np
from PIL import Image

import chainer
from chainer import cuda


def generate_image(gen, row_num, col_num, output_dir, z=None):
    xp = gen.xp
    if z is None:
        z = np.random.random((row_num * col_num, gen.latent_size)).astype(np.float32)
    z = xp.asarray(z)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    @chainer.training.make_extension()
    def make_image(trainer):
        with chainer.using_config('train', False):
            x = gen(z)
        x = cuda.to_cpu(x.data)
        x = (x * 255).clip(0, 255).astype(np.uint8)
        _b, ch, height, width = x.shape
        x = x.reshape((row_num, col_num, ch, height, width))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((row_num * height, col_num * height))
        file_name = 'gen_{0:06d}.png'.format(trainer.updater.iteration)
        Image.fromarray(x).save(os.path.join(output_dir, file_name))
    return make_image
