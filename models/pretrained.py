from torchvision import models as pretrain_models
import torch.nn as nn

PRETRAINED_MODELS = ["alexnet", "squeezenet", "resnet", "vgg"]


def build_pretrained_model(model_name):
    """Returns a pre-trained model given its name"""
    assert model_name in PRETRAINED_MODELS, "Choose model name in {}".format(PRETRAINED_MODELS)
    if model_name == "alexnet":
        model = pretrain_models.alexnet(pretrained=True)
        n_fts = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_fts, 1)
        model.classifier.add_module("7", nn.Sigmoid())
    elif model_name == "squeezenet":
        model = pretrain_models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.add_module("4", nn.Sigmoid())
        model.num_classes = 1
    elif model_name == "resnet":
        model = pretrain_models.resnet18(pretrained=True)
        n_fts = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(n_fts, 1),
                                 nn.Sigmoid())
    elif model_name == "vgg":
        model = pretrain_models.vgg11_bn(pretrained=True)
        n_fts = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_fts, 1)
        model.classifier.add_module("7", nn.Sigmoid())
    return model
