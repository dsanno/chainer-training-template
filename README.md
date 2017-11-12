Chainer Training Template
====

# Usage

## Copy files

Copy files under "template" directory

## Implement some modules

### dataset

* `get_dataset`
    * returns Chainer dataset.

### method

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

### net

Implement your neural network class that extends `chainer.Chain`.

## Implement scripts

Implement your training, prediction, and evaluation script.
