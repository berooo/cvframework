
import sys, os
import numpy as np
import torch
import torch.nn as nn
from graph.normalization import L2N
from graph.builGraph import retrievalNet
from network.multimodal.multinet import GlobalHead

class MultinetExtraction(nn.Module):
    def __init__(self,modelName='resnet50'):
        super(MultinetExtraction,self).__init__()
        self.branch_c = nn.Sequential(*list(retrievalNet(modelName).children())[0][:-1])
        self.branch_p = nn.Sequential(*list(retrievalNet(modelName).children())[0][:-1])
        # self.branch_c = list(list(retrievalNet(modelName).children())[0].children())[0][:-1]
        # self.branch_p = list(list(retrievalNet(modelName).children())[0].children())[0][:-1]
        self.shared = nn.Sequential(*list(retrievalNet(modelName).children())[0][-1])
        self.head = GlobalHead(2048, 2048)
        self.norm = L2N()


    def forward(self,x,mode):

        if mode=='c':
            median_feature = self.branch_c(x)
        elif mode=='p':
            median_feature = self.branch_p(x)

        #mfeature=self.norm(median_feature.view(median_feature.size(0),-1))

        global_feature = self.head(self.shared(median_feature))

        return global_feature

