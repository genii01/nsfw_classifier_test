import torch
import torch.nn as nn
import torchvision.models as models


def create_model(config):
    if config.model.name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=config.model.pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                         config.model.num_classes)
    elif config.model.name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=config.model.pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                         config.model.num_classes)
    return model
