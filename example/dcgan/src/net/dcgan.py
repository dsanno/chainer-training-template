import chainer
from chainer import functions as F
from chainer import links as L


class BatchConv2D(chainer.Chain):

    def __init__(self, input_ch, output_ch, *args, **kwargs):
        super(BatchConv2D, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(input_ch, output_ch, *args, **kwargs)
            self.bn = L.BatchNormalization(output_ch)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.leaky_relu(h)


class BatchDeconv2D(chainer.Chain):

    def __init__(self, input_ch, output_ch, *args, **kwargs):
        super(BatchDeconv2D, self).__init__()
        with self.init_scope():
            self.deconv = L.Deconvolution2D(input_ch, output_ch, *args, **kwargs)
            self.bn = L.BatchNormalization(output_ch)

    def __call__(self, x):
        h = self.deconv(x)
        h = self.bn(h)
        return F.relu(h)


class BatchLinear(chainer.Chain):

    def __init__(self, input_size, output_size, *args, **kwargs):
        super(BatchLinear, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(input_size, output_size, *args, **kwargs)
            self.bn = L.BatchNormalization(output_size)

    def __call__(self, x):
        h = self.fc(x)
        h = self.bn(h)
        return F.relu(h)


class Discriminator(chainer.Chain):

    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = BatchConv2D(1, 16, 4, pad=1, stride=2, initialW=w)
            self.conv2 = BatchConv2D(16, 32, 4, pad=1, stride=2, initialW=w)
            self.conv3 = BatchConv2D(32, 64, 4, pad=1, stride=2, initialW=w)
            self.fc4 = L.Linear(4 * 4 * 64, 1, initialW=w)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        return self.fc4(h)


class Generator(chainer.Chain):

    def __init__(self, latent_size, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__()
        self.latent_size = latent_size
        with self.init_scope():
            self.fc1 = BatchLinear(latent_size, 4 * 4 * 128, initialW=w)
            self.deconv2 = BatchDeconv2D(128, 64, 4, pad=1, stride=2, initialW=w)
            self.deconv3 = BatchDeconv2D(64, 32, 4, pad=1, stride=2, initialW=w)
            self.deconv4 = L.Deconvolution2D(32, 1, 4, pad=1, stride=2, initialW=w)

    def __call__(self, x):
        h = self.fc1(x)
        h = F.reshape(h, (-1, 128, 4, 4))
        h = self.deconv2(h)
        h = self.deconv3(h)
        return self.deconv4(h)
