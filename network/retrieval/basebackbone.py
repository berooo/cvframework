import math
import torch
import torch.nn as nn
from graph.normalization import L2N
from ..utils import getpool,OUTPUT_DIM,Flatten,networks_map


class Base(nn.Module):
    def __init__(self,cfg,pretrained=True):
        super(Base, self).__init__()

        model=networks_map[cfg.MODEL.NAME](pretrained=pretrained)
        self.baselayer=nn.Sequential(*list(model.children())[:-2])
        self.norm = L2N()
        self.pool=getpool(cfg.MODEL.POOL)()
        self.fc=nn.Linear(cfg.MODEL.HEADS.REDUCTION_DIM,cfg.MODEL.NUM_CLASSES,bias=True)
        self.Flatten=Flatten()

    def forward(self, x):

        x=self.baselayer(x)

        x=self.pool(x)
        feat=self.Flatten(x)
        x=self.fc(feat)

        return feat,x


