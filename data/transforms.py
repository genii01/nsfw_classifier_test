import torch
from torchvision import transforms
from PIL import Image


class ChessTransforms:
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (config.data.img_size, config.data.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(config.augmentation.rotation),
                transforms.RandomResizedCrop(
                    config.data.img_size,
                    scale=config.augmentation.scale
                ),
                transforms.ColorJitter(
                    brightness=config.augmentation.brightness,
                    contrast=config.augmentation.contrast,
                    saturation=config.augmentation.saturation,
                    hue=config.augmentation.hue
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (config.data.img_size, config.data.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, img):
        return self.transform(img)
