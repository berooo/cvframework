import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import cv2
import numpy as np
from graph.builGraph import retrievalNet
from graph.normalization import L2N
from graph.model_mapping import networks_map

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class GlobalHead(nn.Module):
    def __init__(self, w_in, nc):
        super(GlobalHead, self).__init__()
        self.pool = GeneralizedMeanPoolingP()
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class multi_net(nn.Module):
    def __init__(self,modelName='resnet50'):
        super(multi_net, self).__init__()

        self.branch_c = nn.Sequential(*list(retrievalNet(modelName).children())[0][:-1])
        self.branch_p=nn.Sequential(*list(retrievalNet(modelName).children())[0][:-1])
        #self.branch_c = list(list(retrievalNet(modelName).children())[0].children())[0][:-1]
        #self.branch_p = list(list(retrievalNet(modelName).children())[0].children())[0][:-1]
        self.shared=nn.Sequential(*list(retrievalNet(modelName).children())[0][-1])
        self.head=GlobalHead(2048,2048)
        self.norm = L2N()

    def forward(self, cx,px):
        median_feature_c = self.branch_c(cx)
        median_feature_p=self.branch_p(px)

        global_feature_c=self.head(self.shared(median_feature_c))
        global_feature_p=self.head(self.shared(median_feature_p))

        return global_feature_c,global_feature_p