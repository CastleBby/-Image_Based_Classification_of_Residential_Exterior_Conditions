"""
Purpose:

--------
Defines the convolutional neural network architecture used for image classification.
This module loads a pretrained ImageNet model and adapts it to the residential
exterior condition classification task.

Inputs:

-------
- num_classes (int): Number of output classes for classification

Outputs:

--------
- model (torch.nn.Module): Configured MobileNetV2 model ready for training

"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes=3):
    model = models.mobilenet_v2(pretrained="DEFAULT")

    # freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model