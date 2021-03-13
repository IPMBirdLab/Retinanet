import torch
from torchvision.transforms import functional as F
import numpy as np


class ToTensor(object):
    def __init__(self, device=None):
        self.device = device

    def __call__(self, image, target):
        if not torch.is_tensor(image):
            image = F.to_tensor(image)

        if isinstance(target, dict):
            for key, value in target.items():
                if not torch.is_tensor(value):
                    target[key] = torch.as_tensor(np.array(value), dtype=torch.int64)
                if self.device is not None:
                    target[key] = target[key].to(self.device)
        elif isinstance(target, np.ndarray) or isinstance(target, list):
            target = torch.tensor(target)
            if self.device is not None:
                target = target.to(self.device)

        if self.device is not None:
            image = image.to(self.device)

        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
