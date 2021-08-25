import os
import logging
import torch
import torchvision
import numpy as np
from torch.utils.data import SubsetRandomSampler, Sampler, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
from imagenet import ImageNet
from transforms.augment_policy import Augmentation
from transforms.augmentations import *
from common.utils import get_logger

logger = get_logger('Augment')
logger.setLevel(logging.INFO)

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_dataloaders(is_ssl, config, dataroot, only_eval):
    # aug = config.get('aug', None) if not only_eval else None
    aug = config.get('aug', None)
    logger.debug('augmentation: %s' % aug)
    if is_ssl:
        return get_dataloaders_ssl(config, dataroot, aug)
    else:
        return get_dataloaders_supervised(config, dataroot, aug)


def get_dataloaders_ssl(config, dataroot, aug):
    dataset, batch = config['dataset'], config['batch']
    batch_unsup = config['batch_unsup']
    apply_cutout = True if (config['dataset'].startswith('cifar') or config['dataset'].startswith('stl')) else False
    print(f'apply cutout for unsupervised images: {apply_cutout}')
    unsup_augmentation = Augmentation(aug,
                                      config.get(aug, None),
                                      config['dataset'],
                                      apply_baseline_aug=True,
                                      apply_cutout=apply_cutout,
                                      is_ssl=True)
    if aug in ['divaug', 'randaug']:
        enlarge = config[aug]['C']
    else:
        enlarge = 1
    aug_sup = Augmentation(aug=None,
                           aug_params=None,
                           dataset=config['dataset'],
                           apply_baseline_aug=True,
                           is_ssl=False,
                           apply_cutout=False)
    if dataset == 'cifar10':
        total_trainset = CIFAR10Dataset(root=dataroot, train=True,
                                        download=True,
                                        transforms_cls=aug_sup)
        unsup_trainset = UnsupervisedCIFAR10(root=dataroot, train=True,
                                             download=True, enlarge=enlarge,
                                             transforms_cls=unsup_augmentation)
        testset = CIFAR10Dataset(root=dataroot, train=False,
                                 download=True,
                                 transforms_cls=None)
        train_size = config.get('train_size', 4000)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        train_sampler = SubsetSampler(train_idx)
        trainloader = torch.utils.data.DataLoader(
            total_trainset, batch_size=batch, shuffle=False,
            num_workers=8, pin_memory=True, sampler=train_sampler, drop_last=True)
    else:
        raise ValueError(f'no support {dataset}')

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch*4, shuffle=False, num_workers=8, pin_memory=True,
        drop_last=False)
    unsuploader = torch.utils.data.DataLoader(
        unsup_trainset, batch_size=batch_unsup, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=False)
    return trainloader, unsuploader, testloader


def get_dataloaders_supervised(config, dataroot, aug):
    dataset, batch = config['dataset'], config['batch']
    if config['dataset'] in ['cifar10', 'cifar100', 'svhn']:
        apply_cutout = True
    else:
        apply_cutout = False
    augmentation = Augmentation(aug,
                                config.get(aug, None),
                                config['dataset'],
                                apply_baseline_aug=True,
                                is_ssl=False,
                                apply_cutout=apply_cutout)

    # ==== Just for getting models without any data augmentation ====
    # augmentation = Augmentation(aug,
    #                             config.get(aug, None),
    #                             config['dataset'],
    #                             apply_baseline_aug=False,
    #                             is_ssl=False,
    #                             apply_cutout=False)
    # ==== Just for getting models without any data augmentation ====
    if aug in ['divaug', 'randaug']:
        enlarge = config[aug].get('C', 1)
    else:
        enlarge = 1
    if dataset == 'cifar10':
        total_trainset = CIFAR10Dataset(root=dataroot, train=True, download=True,
                                        transforms_cls=augmentation,
                                        enlarge=enlarge)
        testset = CIFAR10Dataset(root=dataroot, train=False, download=True,
                                 transforms_cls=None)
    elif dataset == 'cifar100':
        total_trainset = CIFAR100Dataset(root=dataroot, train=True, download=True,
                                         transforms_cls=augmentation,
                                         enlarge=enlarge)
        testset = CIFAR100Dataset(root=dataroot, train=False, download=True,
                                  transforms_cls=None)
    elif dataset == 'imagenet':
        total_trainset = CustomizedImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),
                                            split='train',
                                            download=True,
                                            enlarge=enlarge,
                                            transforms_cls=augmentation)
        testset = CustomizedImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),
                                     split='val',
                                     download=True,
                                     transforms_cls=None)
        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

    elif dataset == 'svhn':
        trainset = CutomizedSVHN(root=dataroot, split='train',
                                 download=True,
                                 enlarge=enlarge,
                                 transforms_cls=augmentation)
        extraset = []
        # extraset = CutomizedSVHN(root=dataroot, split='extra',
        #                          download=True,
        #                          enlarge=enlarge,
        #                          transforms_cls=augmentation)
        # total_trainset = ConcatDataset([trainset, extraset])
        total_trainset = trainset
        testset = CutomizedSVHN(root=dataroot, split='test',
                                download=True, transform=None)
        print(f'len trainset: {len(trainset)}\n'
              f'len extraset: {len(extraset)}\n'
              f'len total trainset; {len(total_trainset)}\n'
              f'len testset: {len(testset)}')
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, pin_memory=True,
        num_workers=16, drop_last=False
    )
    return trainloader, None, testloader


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class CustomizedImageNet(ImageNet):
    num_classes = 1000

    def __init__(self, transforms_cls=None, enlarge=1, **kwargs):
        self.transforms_cls = transforms_cls
        self.enlarge = enlarge
        super().__init__(**kwargs)
        if self.split == 'train':
            self.defaults = [EfficientNetRandomCrop(224),
                             Resize((224, 224)),
                             HorizontalFlip(prob=0.5),
                             ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
                             ToTensor(),
                             Lighting(0.1, _IMAGENET_PCA['eigval'],  _IMAGENET_PCA['eigvec']),
                             Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
        else:
            self.defaults = [EfficientNetCenterCrop(224),
                             Resize((224, 224)),
                             ToTensor(),
                             Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]

    def gen_transforms(self):
        if self.transforms_cls is not None:
            return self.transforms_cls()
        else:
            return []

    def enlarge_augment_imgs(self, img, target):
        imgs = []
        for _ in range(self.enlarge):
            if self.transforms_cls is None:
                transform = self.defaults
            else:
                transform = self.gen_transforms() + self.defaults
            img_, target = Compose(transform)(img, target,
                                              data=self.imgs,
                                              targets=self.targets,
                                              transforms=transform,
                                              num_classes=self.num_classes)
            imgs.append(img_)
        if self.enlarge == 1:
            img = imgs[0]
        else:
            img = torch.stack(imgs, dim=0)
        return img, target

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return self.enlarge_augment_imgs(img, target)


class CutomizedSVHN(torchvision.datasets.SVHN):
    num_classes = 10

    def __init__(self, transforms_cls=None, enlarge=1, return_sampled_trans=False, **kwargs):
        self.transforms_cls = transforms_cls
        self.defaults = [ToTensor(),
                         Normalize(_CIFAR_MEAN, _CIFAR_STD)]
        self.enlarge = enlarge
        self.return_sampled_trans = return_sampled_trans
        super().__init__(**kwargs)
        self.targets = self.labels

    def gen_transforms(self):
        if self.transforms_cls is not None:
            return self.transforms_cls()
        else:
            return []

    def enlarge_augment_imgs(self, img, target):
        imgs = []
        sampled_transforms = []
        for _ in range(self.enlarge):
            if self.transforms_cls is None:
                sampled_transform = []
            else:
                sampled_transform = self.gen_transforms()
            sampled_transforms.append(str(sampled_transform[:2]))
            transform = sampled_transform + self.defaults
            img_, target = Compose(transform)(img, target,
                                              data=self.data,
                                              targets=self.targets,
                                              transforms=transform,
                                              num_classes=self.num_classes)
            imgs.append(img_)
        if self.enlarge == 1:
            img = imgs[0]
        else:
            img = torch.stack(imgs, dim=0)
        if self.return_sampled_trans:
            return img, target, sampled_transforms
        else:
            return img, target

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return self.enlarge_augment_imgs(img, target)


class CustomizedCIFAR10(torchvision.datasets.CIFAR10):
    num_classes = 10

    def __init__(self, transforms_cls=None, enlarge=1, return_sampled_trans=False, **kwargs):
        self.transforms_cls = transforms_cls
        self.defaults = [ToTensor(),
                         Normalize(_CIFAR_MEAN, _CIFAR_STD)]
        self.enlarge = enlarge
        self.return_sampled_trans = return_sampled_trans
        super().__init__(**kwargs)
        # self.targets = F.one_hot(torch.Tensor(self.targets).to(torch.long), num_classes=self.num_classes).float()

    def gen_transforms(self):
        if self.transforms_cls is not None:
            return self.transforms_cls()
        else:
            return []

    def enlarge_augment_imgs(self, img, target):
        imgs = []
        sampled_transforms = []
        for _ in range(self.enlarge):
            if self.transforms_cls is None:
                sampled_transform = []
            else:
                sampled_transform = self.gen_transforms()
            sampled_transforms.append(str(sampled_transform[:2]))
            transform = sampled_transform + self.defaults
            img_, target = Compose(transform)(img, target,
                                              data=self.data,
                                              targets=self.targets,
                                              transforms=transform,
                                              num_classes=self.num_classes)
            imgs.append(img_)
        if self.enlarge == 1:
            img = imgs[0]
        else:
            img = torch.stack(imgs, dim=0)
        if self.return_sampled_trans:
            return img, target, sampled_transforms
        else:
            return img, target


class CIFAR10Dataset(CustomizedCIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return self.enlarge_augment_imgs(img, target)


class UnsupervisedCIFAR10(CustomizedCIFAR10):

    def __init__(self, **kwargs):
        self.transforms_cls_for_ori = [RandomCrop(size=32, padding=4),
                                       HorizontalFlip(prob=0.5)]
        super().__init__(**kwargs)

    def __getitem__(self, index):
        ori_img, target = super().__getitem__(index)
        imgs, _ = self.enlarge_augment_imgs(ori_img.copy(), target)
        transform_for_ori = self.transforms_cls_for_ori + self.defaults
        ori_img, _ = Compose(transform_for_ori)(ori_img, target,
                                                data=self.data,
                                                targets=self.targets,
                                                transforms=transform_for_ori,
                                                num_classes=self.num_classes)
        return ori_img, imgs


class CIFAR100Dataset(CIFAR10Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_classes = 100