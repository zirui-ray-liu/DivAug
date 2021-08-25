import torch.nn.functional as F
from transforms.augmentations import *
from transforms.achieve import _augment_list, searched_polices, augment_dict, _randaug_list
from common.utils import kmean_pp, euclidean_d_1
from common.smooth_ce import SoftCrossEntropyLoss

from kmeanpp import par_kmeanpp

SCE = SoftCrossEntropyLoss(reduction='none')


def augment_list(aug, dataset, is_ssl):
    if aug is None:
        return []
    if is_ssl:
        if dataset.startswith('cifar'):
            l = [
                (ShearX, -0.3, 0.3),  # 0
                (ShearY, -0.3, 0.3),  # 1
                (TranslateX, -0.45, 0.45),  # 2
                (TranslateY, -0.45, 0.45),  # 3
                (Rotate, -30, 30),  # 4
                (AutoContrast, 0, 1),  # 5
                (Invert, 0, 1),  # 6
                (Equalize, 0, 1),  # 7
                (Solarize, 0, 256),  # 8
                (Posterize, 4, 8),  # 9
                (Contrast, 0.1, 1.9),  # 10
                (Color, 0.1, 1.9),  # 11
                (Brightness, 0.1, 1.9),  # 12
                (Sharpness, 0.1, 1.9),  # 13
                (Cutout, 0, 0.2),  # 14
                (FlipLR, 0, 1),  # 15
                (FlipUD, 0, 1),  # 16
                (Blur, 0, 1),  # 17
                (Smooth, 0, 1),  # 18
                (CropBilinear, 0, 0.1)  # 19
            ]
        else:
            l = [(AutoContrast, 0, 1),
                 (Brightness, 0.1, 1.9),  # 12
                 (Color, 0.1, 1.9),  # 11
                 (Contrast, 0.1, 1.9),
                 (Cutout, 0, 0.5),  # 4
                 (Equalize, 0, 1),  # 7
                 (Invert, 0, 1),  # 6
                 (Posterize, 4, 8),  # 9
                 (Rotate, -30, 30),  # 4
                 (Sharpness, 0.1, 1.9),  # 13
                 (ShearX, -0.3, 0.3),  # 0
                 (ShearY, -0.3, 0.3),  # 1
                 (Solarize, 0, 256),  # 8
                 (Smooth, 0, 1),  # 18
                 (TranslateX, -0.3, 0.3),  # 2
                 (TranslateY, -0.3, 0.3),  # 3
                 ]
    # search space of 3 baselines under the supervised setting
    elif aug in [ 'divaug', 'fastaa']:
        l = _augment_list(False)
    elif aug == 'aa':
        l = _augment_list(True)
    elif aug == 'randaug':
        l = _randaug_list()
    else:
        raise NotImplementedError
    return l


def div_augment(config, data, model):
    c, s = config['C'], config['S']
    model.eval()
    with torch.no_grad():
        logits = model(data, return_featmap=False)
        preds = F.softmax(logits, dim=1)
        batch_size = preds.shape[0] // c
        preds_ = preds.reshape(batch_size, c, -1)
        pair_dist = euclidean_d_1(preds_).cpu()
        preds_ = preds_.cpu()
        index = par_kmeanpp(preds_, pair_dist, s).reshape(-1)
    model.train()
    del preds
    return index

# def div_augment(config, data, model):
#     c, s = config['C'], config['S']
#     model.eval()
#     with torch.no_grad():
#         logits = model(data, return_featmap=False)
#         preds = F.softmax(logits, dim=1)
#         batch_size = preds.shape[0] // c
#         preds_ = preds.reshape(batch_size, c, -1)
#         pair_dist = euclidean_d_1(preds_).cpu().numpy()
#         preds_ = np.nan_to_num(preds_.cpu().numpy())
#         index = np.vstack(
#             [
#              kmean_pp(preds_[i], pair_dist[i], s, i*c)
#              for i in range(batch_size)]
#         ).reshape(-1)

#         # batch_size = data.shape[0] // c
#         # index = np.vstack(
#         #     [
#         #      i * c + np.array(random.choices(range(c), k=s))
#         #      for i in range(batch_size)]
#         # ).reshape(-1)

#     model.train()
#     del preds
#     return index


class Augmentation(object):
    def __init__(self, aug, aug_params, dataset,
                 apply_baseline_aug=True,
                 apply_cutout=True,
                 is_ssl=False):
        self.aug = aug
        self.aug_params = aug_params
        self.is_ssl = is_ssl
        self.augment_list = augment_list(aug, dataset, is_ssl)
        self.dataset = dataset
        self.apply_cutout = apply_cutout
        self.apply_baseline_aug = apply_baseline_aug

    def __call__(self):
        parsed_transforms = []
        if self.aug is None:
            parsed_transforms = []
        elif self.aug == 'randaug':
            if self.is_ssl:
                level = random.random()
                prob = 0.5
            else:
                level = float(self.aug_params['M']) / 30
                prob = 1.0
            ops = random.choices(self.augment_list, k=self.aug_params['N'])
            for op_cls, min_val, max_val in ops:
                op = op_cls(value_range=(min_val, max_val), prob=prob, level=level)
                parsed_transforms.append(op)
        elif self.aug == 'divaug':
            ops = random.choices(self.augment_list, k=self.aug_params['N'])
            for op_cls, min_val, max_val in ops:
                prob = random.random()
                level = random.random()
                op = op_cls(value_range=(min_val, max_val), prob=prob,
                            level=level)
                parsed_transforms.append(op)
        elif self.aug in ['fastaa', 'aa']:
            if self.dataset != 'cifar10':
                raise ValueError('fast AutoAugment and AutoAugment only supported for CIFAR-10')
            policy_pools = searched_polices(self.aug)
            sub_policy = random.choice(policy_pools)
            for name, prob, level in sub_policy:
                op_cls, min_val, max_val = augment_dict[name]
                op = op_cls(value_range=(min_val, max_val), prob=prob,
                            level=level)
                parsed_transforms.append(op)

        if self.apply_baseline_aug:
            if self.dataset.startswith('cifar'):
                size = 32
                parsed_transforms += [RandomCrop(size=size, padding=4),
                                      HorizontalFlip(prob=0.5)]
            elif self.dataset == 'svhn':
                parsed_transforms += [RandomCrop(size=32, padding=4)]
            elif self.dataset == 'imagenet':
                pass
            else:
                raise NotImplementedError
        if self.apply_cutout:
            parsed_transforms += [Cutout(prob=1.0, level=1.0, value_range=(0, 0.5))]
        return parsed_transforms
