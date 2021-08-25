import torch

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from .resnet import ResNet
from .pyramidnet import PyramidNet
from .shakeshake.shake_resnet import ShakeResNet
from .wideresnet import WideResNet
from .shakeshake.shake_resnext import ShakeResNeXt


def get_model(conf, dataset, num_class=10):
    name = conf['type']
    if name == 'resnet18':
        model = ResNet(dataset=dataset, depth=18, num_classes=num_class, bottleneck=True)
    elif name == 'resnet50':
        model = ResNet(dataset=dataset, depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset=dataset, depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_2':
        model = WideResNet(28, 2, dropout_rate=0.0, num_classes=num_class)

    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_class)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_class)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)

    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)

    elif name == 'pyramid':
        model = PyramidNet('cifar10', depth=conf['depth'], alpha=conf['alpha'], num_classes=num_class, bottleneck=conf['bottleneck'])
    else:
        raise NameError('no model named, %s' % name)

    model = model.cuda()
    model = DataParallel(model)
    cudnn.benchmark = True
    cudnn.enabled = True
    return model


def get_num_class(dataset):
    return {
        'cifar10': 10,
        'cifar100': 100,
        'svhn': 10,
        'imagenet': 1000,
    }[dataset]
