Chainer Training Template
====

# Usage

## Copy files

Copy files under "template" directory

## Implement some modules

### dataset (src/dataset.py)

* `get_dataset`
    * returns dictionary of Chainer datasets.
        * Dictionary keys must include "train" for training dataset.
        * Dictionary keys must include "test" for test dataset if you need.
        * Dictionary keys may include "validation" for validation while training.
    * example:
    ```
    def get_dataset():
        train, test = chainer.datasets.get_mnist()
        validation, train = chainer.datasets.split_dataset_random(train, 5000)
        return {
            'train': train,
            'validation': validation,
            'test': test,
        }
    ```

### model (src/model.py)

* `calculate_metrics`
    * returns loss and other metrics for each optimizer.
    * example:
    ```
    def calculate_metrics(net, batch):
        x, t = batch
        y = net(x)
        loss = F.softmax_cross_entropy
        return loss
    ```
    with other metorics:
    ```
    def calculate_metrics(net, batch):
        x, t = batch
        y = net(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    ```
    with two optimizers:
    ```
    def calculate_metrics(nets, batch):
        gen = nets['gen']
        dis = nets['dis']

        # ... calculate loss
        loss_gen = ...
        loss_dis = ...

        return {
            'gen': loss_gen,
            'dis': loss_dis,
        }
        # or
        return {
            'gen': {},
            'dis': loss_dis,
        }
    ```
* `make_eval_func`:
    * makes evaluation function for Chainer `Evaluator` extension.
* `predict`:
    * returns prediction of your neural network.

### net (src/net/network_name.py)

Implement your neural network class that extends `chainer.Chain`.

## Implement scripts

### Training script (src/train.py)

Usually you don't have to modify training script except adding chainer training
extensions if you need.

### Prediction script (src/predict.py)

Usually you don't have to modify prediction script.

### Evaluation script (src/evaluate.py)

Implement your evaluation script.

## Configuration files

### Application configuration (config/app.json)

Configuration independent of each training/prediction.

* `train`(object): training configuration.
    * `dump_graph`(str): Name of the root for `dump_graph` extension.
    * `plot_report`(object):
        * key: File name for figure file without extension.
        * value(array of str): `y_keys` for `PlotReport` extension.
    * `print_report`(array of str): `entries` for `PrintReport` extension.
    * `max_value_trigger`(string or null): The best model is saved when the value associated with this key string becomes maximum. (e.g. "validation/main/accuracy")
    * `min_value_trigger`(string or null): The best model is saved when the value associated with this key string becomes minimum. (e.g. "validation/main/loss") You can use only either `max_value_trigger` or `min_value_trigger`.

### Configuration (config/config.json)

Configuration depends on each training/prediction.

* `batch_size`(int): Mini batch size for training/prediction.
* `epoch`(int): The number of epochs for training.
* `gpu`(int): Default GPU device index (negative value indicates CPU). This can be changed with command line option.
* `dataset`(object):
    * `parameter`(object, array, single value or null): Parameter for `get_dataset`. `null` indicates no parameter.
* `network`(object): Network information.
    * key: network identifier.
    * value(object):
        * `class`(str): Network class.
        * `parameter`(object, array, single value or null): Parameter for network constructor.
        * `optimizer`(object): Network optimizer information.
            * `class`(str): Optimizer class.
            * `parameter`(object, array, single value or null): Parameter for optimizer constructor.
            * `hook`(array of objects):
                * `class`(str): Hook class.
                * `parameter`(object, array, single value or null): Parameter for hook constructor.
* `output_dir`(str): Output directory path for training.
