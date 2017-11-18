from importlib import import_module

import chainer


def _load_class(class_path, default_package=None):
    parts = class_path.split('.')
    package_path = '.'.join(parts[:-1])
    class_name = parts[-1]
    if default_package is not None and hasattr(default_package, class_name):
        return getattr(default_package, class_name)
    return getattr(import_module(package_path), class_name)


def _create_instance(class_path, parameter=None, default_package=None):
    constructor = _load_class(class_path, default_package)
    if isinstance(parameter, list):
        return constructor(*parameter)
    elif isinstance(parameter, dict):
        return constructor(**parameter)
    elif parameter is not None:
        return constructor(parameter)
    return constructor()


def create_network(params):
    parameter = params.get('parameter', None)
    net = _create_instance(params['class'], parameter)
    return net


def create_optimizer(params, net):
    parameter = params.get('parameter', None)
    optimizer = _create_instance(params['class'], parameter, chainer.optimizers)
    optimizer.setup(net)
    for param in params.get('hook', []):
        parameter = param.get('parameter', None)
        hook = _create_instance(param['class'], parameter, chainer.optimizer)
        optimizer.add_hook(hook)
    return optimizer
