import numpy as np


import chainer
from chainer import cuda
from chainer import functions as F
from chainer.dataset import convert


def calculate_metrics(net, batch):
    x, t = batch
    y = net(x)
    loss = F.softmax_cross_entropy(y, t)
    accuracy = F.accuracy(y, t)
    return {
        'loss': loss,
        'accuracy': accuracy,
    }


def make_eval_func(net):
    def evaluate(*in_arrays):
        metrics = calculate_metrics(net, in_arrays)
        chainer.report(metrics, net)
    return evaluate


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
