import numpy as np

import chainer
from chainer import cuda
from chainer import functions as F
from chainer.dataset import convert


def predict(net, x):
    y = net(x)
    return F.softmax(y)


def predict_dataset(net, iterator, converter=convert.concat_examples, device=None):
    scores = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for batch in iterator:
            x, t = converter(batch, device)
            y = predict(net, x)
            scores.append(cuda.to_cpu(y.data))
    return np.concatenate(scores, axis=0)
