import math
import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import torch
import torchvision
import numpy as np
from torchvision.datasets.folder import default_loader
from PIL import Image


def level2val(level, value_range, val_type='float'):
    v = value_range[0] + level * float(value_range[1] - value_range[0])
    return v


class Transform(object):

    def __init__(self, value_range=None, name=None, prob=1.0, level=0):
        self.name = name if name is not None else type(self).__name__
        self.prob = prob
        if level < 0 or level > 1:
            raise ValueError('level must be in [0, 1]')
        self.level = level
        self.value_range = value_range

    def transform(self, img, label, **kwargs):
        return img, label

    def __call__(self, img, label, **kwargs):
        if random.random() <= self.prob:
            return self.transform(img, label, **kwargs)
        else:
            return img, label

    def __repr__(self):
        return f'<Transform ({self.name}, prob={self.prob}, level={self.level})>'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label, **kwargs):
        for idx, t in enumerate(self.transforms):
            kwargs['idx'] = idx
            img, label = t(img, label, **kwargs)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ShearX(Transform):
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), label


class ShearY(Transform):
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), label


class TranslateX(Transform):  # [-150, 150] => percentage: [-0.45, 0.45]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        if random.random() > 0.5:
            v = -v
        v = v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), label


class TranslateXabs(Transform):  # [-150, 150] => percentage: [-0.45, 0.45]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), label


class TranslateY(Transform):  # [-150, 150] => percentage: [-0.45, 0.45]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        if random.random() > 0.5:
            v = -v
        v = v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), label


class TranslateYabs(Transform):  # [-150, 150] => percentage: [-0.45, 0.45]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), label


class Rotate(Transform):  # [-30, 30]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        if random.random() > 0.5:
            v = -v
        return img.rotate(v), label


class AutoContrast(Transform):
    def transform(self, img, label, **kwargs):
        return PIL.ImageOps.autocontrast(img), label


class Invert(Transform):
    def transform(self, img, label, **kwargs):
        return PIL.ImageOps.invert(img), label


class Equalize(Transform):
    def transform(self, img, label, **kwargs):
        return PIL.ImageOps.equalize(img), label


class FlipLR(Transform):  # not from the paper
    def transform(self, img, label, **kwargs):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label


class FlipUD(Transform):  # not from the paper
    def transform(self, img, label, **kwargs):
        return img.transpose(Image.FLIP_TOP_BOTTOM), label


class Blur(Transform): # not from the paper
    def transform(self, img, label, **kwargs):
        return img.filter(PIL.ImageFilter.BLUR), label


class Smooth(Transform): # not from the paper
    def transform(self, img, label, **kwargs):
        return img.filter(PIL.ImageFilter.SMOOTH), label


class CropBilinear(Transform):
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        v = v * img.size[0]
        size0, size1 = img.size[0], img.size[1]
        cropped = img.crop((v, v, size0 - v, size1 - v))
        resized = cropped.resize((size0, size1), Image.BILINEAR)
        return resized, label


class Flip(Transform):  # not from the paper
    def transform(self, img, label, **kwargs):
        return PIL.ImageOps.mirror(img), label


class Solarize(Transform):  # [0, 256]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        return PIL.ImageOps.solarize(img, v), label


class SolarizeAdd(Transform):
    def transform(self, img, label, addition=0, threshold=128):
        img_np = np.array(img).astype(np.int)
        img_np = img_np + addition
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        return PIL.ImageOps.solarize(img, threshold), label


class Posterize(Transform):  # [4, 8]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        v = int(v)
        v = max(1, v)
        return PIL.ImageOps.posterize(img, v), label


class Contrast(Transform):  # [0.1,1.9]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        return PIL.ImageEnhance.Contrast(img).enhance(v), label


class Color(Transform):  # [0.1,1.9]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        return PIL.ImageEnhance.Color(img).enhance(v), label


class Brightness(Transform):  # [0.1,1.9]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        return PIL.ImageEnhance.Brightness(img).enhance(v), label


class Sharpness(Transform):  # [0.1,1.9]
    def transform(self, img, label, **kwargs):
        v = level2val(self.level, self.value_range)
        return PIL.ImageEnhance.Sharpness(img).enhance(v), label


class Cutout(Transform):  # [0, 60] => percentage: [0, 0.2]

    def transform(self, img, label, **kwargs):
        img = img.copy()
        v = level2val(self.level, self.value_range)
        if v <= 0.:
            return img
        v = v * img.size[0]
        width, height = img.size
        x0 = np.random.uniform(width)
        y0 = np.random.uniform(height)

        x0 = int(max(0, x0 - v / 2.0))
        y0 = int(max(0, y0 - v / 2.0))
        x1 = min(width, x0 + v)
        y1 = min(height, y0 + v)

        xy = (x0, y0, x1, y1)

        if img.mode == "RGB":
            color = (125, 123, 114)
        elif img.mode == "L":
            color = 121
        else:
            raise ValueError(f"Unspported image mode {img.mode}")
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img, label


class Identity(Transform):
    def transform(self, img, label, **kwargs):
        return img, label


class ToTensor(Transform):
    def transform(self, img, label, **kwargs):
        return torchvision.transforms.ToTensor()(img), label


class Lighting(Transform):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec, **kwargs):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)
        super().__init__(**kwargs)

    def transform(self, img, label, **kwargs):
        if self.alphastd == 0:
            return img, label

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img)), label


class Normalize(Transform):
    def __init__(self, mean, std, **kwargs):
        self.mean = mean,
        self.std = std
        super(Normalize, self).__init__(**kwargs)
        self.normalize_func = torchvision.transforms.Normalize(mean, std)

    def transform(self, img, label, **kwargs):
        return self.normalize_func(img), label


class EfficientNetCenterCrop(Transform):
    def __init__(self, imgsize, **kwargs):
        self.imgsize = imgsize
        super(EfficientNetCenterCrop, self).__init__(**kwargs)

    def transform(self, img, label, **kwargs):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)
        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short
        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)), label


class EfficientNetRandomCrop(Transform):
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3),
                 area_range=(0.08, 1.0), max_attempts=10, **kwargs):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize, prob=1.0)
        super(EfficientNetRandomCrop, self).__init__(**kwargs)

    def transform(self, img, label, **kwargs):
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img, label, **kwargs)

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height)), label

        return self._fallback(img, label, **kwargs)


class RandomResizeCrop(Transform):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, **kwargs):
        self.transform_func = torchvision.transforms.RandomResizedCrop(size, scale, ratio, interpolation)
        super().__init__(**kwargs)

    def transform(self, img, label, **kwargs):
        return self.transform_func(img), label


class Resize(Transform):
    def __init__(self, size, interpolation=Image.BILINEAR, **kwargs):
        self.transform_func = torchvision.transforms.Resize(size, interpolation)
        super().__init__(**kwargs)

    def transform(self, img, label, **kwargs):
        return self.transform_func(img), label


class ColorJitter(Transform):
    def __init__(self, brightness, contrast, saturation, **kwargs):
        super().__init__(**kwargs)
        self.transform_func = torchvision.transforms.ColorJitter(brightness, contrast, saturation)

    def transform(self, img, label, **kwargs):
        return self.transform_func(img), label


class RandomCrop(Transform):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant', **kwargs):
        self.transform_func = torchvision.transforms.RandomCrop(size, padding,
                                                                pad_if_needed, fill,
                                                                padding_mode)
        super().__init__(**kwargs)

    def transform(self, img, label, **kwargs):
        return self.transform_func(img), label


class HorizontalFlip(Transform):
    def transform(self, img, label, **kwargs):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label


class SamplePair(Transform):
    def __init__(
        self,
        value_range=None,
        name=None,
        prob=1.0,
        level=0,
        alpha=1.0,
        same_class_ratio=-1.0,
        prob_label=False,
    ):
        self.alpha = alpha
        self.same_class_ratio = same_class_ratio
        self.prob_label = prob_label

        super().__init__(value_range, name, prob, level)

    def transform(self, img, label, **kwargs):
        data = kwargs['data']
        targets = kwargs['targets']
        transforms = kwargs["transforms"]
        num_classes = kwargs["num_classes"]
        if self.alpha > 0.0:
            mix_ratio = np.random.beta(self.alpha, self.alpha)
        else:
            mix_ratio = 1.0

        tot_cnt = len(data)
        idx = np.random.randint(tot_cnt)

        if self.same_class_ratio >= 0:
            same_class = True if np.random.rand() <= self.same_class_ratio else False
            for i in np.random.permutation(tot_cnt):
                if same_class == torch.equal(targets[i], label):
                    idx = i
                    break

        # Calc all transforms before SamplePair
        prev_transforms = transforms[: kwargs["idx"]]

        # Apply all prev SamplePair transforms
        if isinstance(data[idx], tuple):
            cand_img_path = data[idx][0]
            cand_img = default_loader(cand_img_path)
        else:
            cand_data = data[idx]
            if cand_data.shape == (3, 32 , 32):
                cand_data = np.transpose(cand_data, (1, 2, 0))
            cand_img = Image.fromarray(cand_data)
        cand_img, cand_label = Compose(prev_transforms)(
                cand_img, targets[idx], **kwargs
            )

        # sp_img = torchvision.transforms.ToPILImage()(
        #     mix_ratio * torchvision.transforms.ToTensor()(img) +
        #     (1 - mix_ratio) * torchvision.transforms.ToTensor()(cand_img)
        # )
        sp_img = PIL.Image.blend(img, cand_img.resize(size=img.size), alpha=1 - mix_ratio)

        if label is not None:
            if self.prob_label:
                sp_label = mix_ratio * label + (1 - mix_ratio) * cand_label
            else:
                sp_label = label if np.random.random() < mix_ratio else cand_label
        else:
            sp_label = label

        return sp_img, sp_label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"alpha={self.alpha}, same_class_ratio={self.same_class_ratio}, "
            f"prob_label={self.prob_label}>"
        )
