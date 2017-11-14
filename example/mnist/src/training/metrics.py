import chainer
from chainer import functions as F


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
