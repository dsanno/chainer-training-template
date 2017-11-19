import numpy as np


import chainer
from chainer import cuda
from chainer import functions as F
from chainer.dataset import convert


def calculate_metrics(nets, batch):
    gen = nets['gen']
    dis = nets['dis']
    x, t = batch
    xp = cuda.get_array_module(x)
    batch_size = len(x)
    z = xp.asarray(np.random.random((batch_size, gen.latent_size)).astype(np.float32))

    x_fake = gen(z)
    y_real = dis(x)
    y_fake = dis(x_fake)

    real_label = xp.ones((batch_size, 1), dtype=np.int32)
    fake_label = xp.zeros((batch_size, 1), dtype=np.int32)

    loss_dis = F.sigmoid_cross_entropy(y_real, real_label)
    loss_dis += F.sigmoid_cross_entropy(y_fake, fake_label)
    loss_gen = F.sigmoid_cross_entropy(y_fake, real_label)

    return {
        'gen': loss_gen,
        'dis': loss_dis
    }
