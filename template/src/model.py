import numpy as np

import chainer
from chainer import cuda
from chainer import functions as F
from chainer.dataset import convert


def calculate_metrics(net, batch):
    # TODO: Implement your metrics function
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
    # TODO: Implement your prediction function
    y = net(x)
    return F.softmax(y)
