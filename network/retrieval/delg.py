from datasets.imageListDateset import ImagesFromList
from graph.model_mapping import networks_map
import numpy as np
import torch
import torch.nn as nn
import os

from graph.function import ApplyPcaAndWhitening
from util.array_tool import tonumpy,totensor
from collections import OrderedDict
from util.Balancedparallel import DataParallelModel
import torch.nn.functional as F
from network.outputdim import OUTPUT_DIM
from graph.pooling import *

from util.util import *
from losses.arcface_loss import ArcMarginLoss
from network.retrieval.GLEMweight import GLEMweightNet,GLEMweightExtractor

class WeightedSum2d(nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()
    def forward(self, x):
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
                'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, weights.size)
        y = x * weights                                       # element-wise multiplication
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))      # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1
    def __repr__(self):
        return self.__class__.__name__


class Attention2d(nn.Module):

    def __init__(self, in_c, act_fn='relu'):
        super(Attention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)  # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)  # 1x1 conv
        self.softplus = nn.Softplus(beta=1, threshold=20)  # use default setting.
        self.norm = L2N()

    def forward(self, inputs):
        '''
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score
        '''
        x = self.conv1(inputs)
        x = self.act1(x)
        score = self.conv2(x)
        prob = self.softplus(x)

        inputs=self.norm(inputs)
        y=inputs*prob
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))
        feat=torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)

        return feat,prob,score

    def __repr__(self):
        return self.__class__.__name__

class Delg(nn.Module):
    def __init__(self,model,use_pca=True,finetune=False, pool='gem',embedding_layer=True,**kwargs):
        super(Delg, self).__init__()
        self.backbone = kwargs['name']
        numclasses = kwargs['num_classes']
        inchannels = 1000
        if self.backbone.startswith('resnet'):
            if finetune:
                self._baselayer = list(model.children())[0]
            else:
                self._baselayer= nn.Sequential(*list(model.children())[:-2])
            inchannels = OUTPUT_DIM[self.backbone]

        if pool=='gem':
         self.pool=GeMmp()
        elif pool=='spoc':
         self.pool=SPoC()
        elif pool=='scale_avg':
         self.pool = scale_avg_pool()
        elif pool=='vlad':
         self.pool=VLAD()
        else:
         self.pool=MAC()

        if embedding_layer:
            self.embedding_linear=nn.Linear(inchannels,inchannels,bias=True)
        else:
            self.embedding_linear=None

        self.norm=L2N()


        self.attn_classification = nn.Linear(inchannels, numclasses)

        self.desc_classification=ArcMarginLoss(out_features=numclasses)

        self.Flatten = Flatten()
        self.features=OrderedDict()



    def forward(self,x,need_feature=True):

        for name,module in self._baselayer._modules.items():
            x=module(x)
            self.features[name]=x

        x=self.pool(x)
        self.features['pool'] = x
        x=self.Flatten(x)
        x=self.embedding_linear(x)


        return x

if __name__=='__main__':
    qloader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=['/home/shibaorong/modelTorch/test.jpg'], imsize=224),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    kargs = {}
    kargs['name'] = 'resnet50'
    kargs['num_classes'] = 11
    initmodel=networks_map['resnet50'](pretrained=True);
    re=Delg(initmodel,**kargs)

    for input in qloader:
        out=re(input)
        print(out)