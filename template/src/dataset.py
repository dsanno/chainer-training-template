import numpy as np
from chainer import datasets


def _transform(in_data):
    img, label = in_data
    left, top = np.random.randint(0, 5, 2)
    x = np.reshape(img, (28, 28))
    x = np.pad(x, ((2, 2), (2, 2)), 'constant')
    return np.ravel(x[top:top + 28, left:left + 28]), label


def get_dataset():
    # TODO: Implement your dataset
    train, test = datasets.get_mnist()
    validation, train = datasets.split_dataset_random(train, 5000)
    train = datasets.TransformDataset(train, _transform)
    return {
        'train': train,
        'validation': validation,
        'test': test,
    }
