import NETC1
from torchvision import models
import torch.nn as nn
import VGG16
def get_netc_model(name):
    if name=="custom":
        return NETC1.netC()
    elif name=="resnet":
        pass


    elif name=="vgg16":
        model=VGG16.VGG16()
        return model