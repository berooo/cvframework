from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from network import inception, resnet,densenet,dshNet,vgg,resnet_ibn
from network.landmark import GLEM
from network.retrieval import delf

networks_map = {
    'inception_v3': inception.inception_v3,
    'resnet50': resnet.resnet50,
    'resnet18': resnet.resnet18,
    'resnet34':resnet.resnet34,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
    'densenet121':densenet.densenet121,
    'densenet169':densenet.densenet169,
    'densenet161':densenet.densenet161,
    'densenet201':densenet.densenet201,
    'GLEM':GLEM.Network,
    'dshNet':dshNet.dshNet,
    'delf':delf.Delf_V1,
    'vgg16':vgg.vgg16,
    'resnet18_ibn_a':resnet_ibn.resnet18_ibn_a,
    'resnet18_ibn_b':resnet_ibn.resnet18_ibn_b
}

