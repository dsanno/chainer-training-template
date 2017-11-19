import numpy as np
from chainer import datasets


def _transform(in_data, augment=False):
    img, label = in_data
    if augment:
        left, top = np.random.randint(0, 5, 2)
    else:
        left, top = 2, 2
    x = np.reshape(img, (1, 28, 28))
    x = np.pad(x, ((0, 0), (left, 4 - left), (top, 4 - top)), 'constant')
    return x, label


def get_dataset():
    train, test = datasets.get_mnist()
    train = datasets.TransformDataset(train, lambda x: _transform(x, True))
    test = datasets.TransformDataset(train, _transform)
    return {
        'train': train,
        'test': test,
    }
