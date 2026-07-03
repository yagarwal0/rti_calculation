"""
models — network builders for the pruning experiment.

All builders take `num_classes` and return a standard nn.Module made of
nn.Conv2d / nn.Linear / nn.BatchNorm2d layers (3x32x32 inputs), so they work
for CIFAR-10 (10), CIFAR-100 (100) and Tiny ImageNet resized to 32x32 (200).

    from models import build_model, MODELS
    net = build_model('resnet20', num_classes=10)
"""

from .resnet20 import resnet20
from .resnet_8x import ResNet18_8x
from .vgg import VGG


def resnet18(num_classes=10):
    return ResNet18_8x(num_classes=num_classes)


def vgg19(num_classes=10):
    return VGG(config="VGG19", num_classes=num_classes)


# name -> builder(num_classes)
MODELS = {
    'resnet20': resnet20,
    'resnet18': resnet18,
    'vgg19': vgg19,
}


def build_model(name, num_classes):
    name = name.lower()
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}.")
    return MODELS[name](num_classes=num_classes)


__all__ = ["build_model", "MODELS", "resnet20", "resnet18", "vgg19",
           "ResNet18_8x", "VGG"]
