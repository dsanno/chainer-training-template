from __future__ import print_function
import argparse
import json
import numpy as np
import os
import scikitplot as skplt
import six
from matplotlib import pyplot as plt
from PIL import Image


import dataset


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST evaluation example')
    parser.add_argument('prediction_path',
                        help='Prediction file path')
    parser.add_argument('output_dir',
                        help='Result output directory')
    return parser.parse_args()


def sort_prediction(prediction, flag_true, flag_pred):
    flag = np.logical_and(flag_true, flag_pred)
    index = np.arange(len(prediction))[flag]
    y = prediction[flag]
    top_index = y.max(axis=1).argsort()[::-1]
    return index[top_index]


def main():
    args = parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _train, _valid, test = dataset.get_dataset()
    test_label = [t for x, t in test[:]]
    test_label = np.asarray(test_label, dtype=np.int32)
    y = np.load(args.prediction_path)
    predict_label = np.argmax(y, axis=1)

    skplt.metrics.plot_confusion_matrix(test_label, np.argmax(y, axis=1))
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    log_path = os.path.join(output_dir, 'log.txt')
    with open(log_path, 'w') as f:
        accuracy = float(np.sum(test_label == predict_label)) / len(test_label)
        f.write('Acciracy: {}\n'.format(accuracy))

        f.write('\nCorrect Images:\n')
        image_dir = os.path.join(output_dir, 'correct_image')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i in six.moves.range(10):
            rank = sort_prediction(y, test_label == i, predict_label == i)
            for j in rank[:10]:
                scores = ','.join(map(str, y[j]))
                f.write('{0},{1},{1},{2}\n'.format(j, i, scores))
                file_name = '{0}_{1:05d}.png'.format(i, j)
                x = (test[j][0] * 255).astype(np.uint8).reshape((28, 28))
                Image.fromarray(x).save(os.path.join(image_dir, file_name))
        f.write('\nIncorrect Images:\n')
        image_dir = os.path.join(output_dir, 'incorrect_image')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i in six.moves.range(10):
            rank = sort_prediction(y, test_label == i, predict_label != i)
            for j in rank[:10]:
                label = np.argmax(y[j])
                scores = ','.join(map(str, y[j]))
                f.write('{0},{1},{2},{3}\n'.format(j, i, label, scores))
                file_name = '{0}_{1}_{2:05d}.png'.format(i, label, j)
                x = (test[j][0] * 255).astype(np.uint8).reshape((28, 28))
                Image.fromarray(x).save(os.path.join(image_dir, file_name))


if __name__ == '__main__':
    main()
