import chainer
from chainer import functions as F
from chainer import links as L

# TODO: Implement your network (rename file and class if necessary)
class MLP(chainer.Chain):

    def __init__(self, input_size=28 * 28, unit_sizes=[100, 100], output_size=10):
        super(MLP, self).__init__()
        self.layers = []
        with self.init_scope():
            sizes = zip([input_size] + unit_sizes, unit_sizes + [output_size])
            for i, (n, m) in enumerate(sizes):
                link = L.Linear(n, m)
                setattr(self, 'l{}'.format(i + 1), link)
                self.layers.append(link)

    def __call__(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        return self.layers[-1](h)
