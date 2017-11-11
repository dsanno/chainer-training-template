import chainer
from chainer.dataset import convert
from chainer.training import StandardUpdater


def fetch(iterators, converter, device=None):
    return converter(iterators['main'].next(), device)


class TrainingStep(StandardUpdater):


    def __init__(self, iterator, optimizer, step_func,
                 converter=convert.concat_examples, device=None,
                 fetch_func=fetch, step_option=None):
        super(TrainingStep, self).__init__(iterator, optimizer, converter,
                                           device)
        if isinstance(optimizer, dict):
            self._target = {name: opt.terget for name, opt in optimizer.items()}
        else:
            self._target = optimizer.target
        self._fetch_func = fetch_func
        self._step_func = step_func
        self._step_option = step_option

    def update_core(self):
        batch = self._fetch_func(self._iterators, self.converter, self.device)
        if self._step_option is None:
            step_result = self._step_func(self._target, batch)
        else:
            step_result = self._step_func(self._target, batch, self._step_option)
        if isinstance(self._target, dict):
            for name in self._target.keys():
                if step_result.has_key(name):
                    self._post_process(name, step_result['name'])
        else:
            self._post_process('main', step_result)

    def _post_process(self, name, result):
        if isinstance(result, dict):
            loss = result['loss']
            metrics = result
        else:
            loss = result
            metrics = {'loss': loss}
        optimizer = self.get_optimizer(name)
        target = optimizer.target

        target.cleargrads()
        loss.backward()
        optimizer.update()

        chainer.report(metrics, target)
