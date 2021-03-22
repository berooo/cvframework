from graph.pooling import GeMmp, SPoC, scale_avg_pool, VLAD, MAC

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
import torch.nn as nn
from . import inception,resnet,densenet,dshNet,vgg,resnet_ibn,resnest
from .landmark import GLEM


networks_map = {
    'inception_v3': inception.inception_v3,
    'resnet50': resnet.resnet50,
    'resnet18': resnet.resnet18,
    'resnet34':resnet.resnet34,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
    'resnet200':resnest.resnest200,
    'resnet269':resnest.resnest269,
    'densenet121':densenet.densenet121,
    'densenet169':densenet.densenet169,
    'densenet161':densenet.densenet161,
    'densenet201':densenet.densenet201,
    'GLEM':GLEM.Network,
    'dshNet':dshNet.dshNet,
    'vgg16':vgg.vgg16,
    'resnet18_ibn_a':resnet_ibn.resnet18_ibn_a,
    'resnet18_ibn_b':resnet_ibn.resnet18_ibn_b
}


OUTPUT_DIM = {
    'alexnet'               :  256,
    'vgg11'                 :  512,
    'vgg13'                 :  512,
    'vgg16'                 :  512,
    'vgg19'                 :  512,
    'resnet18'              :  512,
    'resnet34'              :  512,
    'resnet50'              : 2048,
    'resnet101'             : 2048,
    'resnet152'             : 2048,
    'densenet121'           : 1024,
    'densenet169'           : 1664,
    'densenet201'           : 1920,
    'densenet161'           : 2208, # largest densenet
    'squeezenet1_0'         :  512,
    'squeezenet1_1'         :  512,
    'dshNet':500
}


def getpool(pool):
    if pool == 'gem':
        return GeMmp
    elif pool == 'spoc':
        return SPoC
    elif pool == 'scale_avg':
        return scale_avg_pool
    elif pool == 'vlad':
        return VLAD
    elif pool == 'mac':
        return MAC

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, input):
    return input.view(input.size(0), -1)