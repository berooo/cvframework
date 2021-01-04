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
from network.retrieval.GLEMweight import GLEMweightNet,GLEMweightExtractor

class Net(nn.Module):
    def __init__(self,model, **kwargs):
        super(Net,self).__init__()
        self.backbone=kwargs['name']
        numclasses=kwargs['num_classes']
        inchannels=1000
        if self.backbone.startswith('resnet'):
            self._baselayer= nn.Sequential(*list(model.children())[:-1])
            inchannels=OUTPUT_DIM[self.backbone]
        self.features=OrderedDict()
        self._linearlayer=nn.Linear(inchannels,numclasses)
        self.Flatten=Flatten()

    def forward(self,x,type='cls'):

        for name,module in self._baselayer._modules.items():
            x=module(x)
            '''self.features[name]=x'''
        #x=self._baselayer(x)
        out=self.Flatten(x)
        if type!='cls':
            return out
        out=self._linearlayer(out)
        return out

class multibranchNet(nn.Module):
    def __init__(self, model, **kwargs):
        super(multibranchNet, self).__init__()
        self.backbone = kwargs['name']
        numclasses = kwargs['num_classes']
        inchannels = 1000
        if self.backbone.startswith('resnet'):
            self._baselayer = nn.Sequential(*list(model.children())[:-1])
            inchannels = OUTPUT_DIM[self.backbone]
        self.features = OrderedDict()
        self.pool = SPoC()
        self.norm = L2N()

        self._linearlayer = nn.Linear(inchannels, numclasses)
        self.Flatten = Flatten()

    def forward(self, x,addBN=False):

        for name, module in self._baselayer._modules.items():
            x = module(x)
            self.features[name] = x

        resdict={}
        out1 = self.Flatten(x)
        resdict['intersect']=out1

        out1 = self._linearlayer(out1)

        resdict['out']=out1

        out2 = self.pool(x)
        self.features['pool'] = out2
        out2 = self.norm(out2)
        self.features['norm'] = out2
        out2 = self.Flatten(out2)

        resdict['feature']=out2

        return resdict

class extractor(nn.Module):
    def __init__(self, model, use_pca=True,pool='spoc', **kwargs):
        super(extractor, self).__init__()
        self.backbone = kwargs['name']
        numclasses = kwargs['num_classes']

        if 'pool' in kwargs:
            pool=kwargs['pool']

        inchannels = 1000
        if self.backbone.startswith('resnet'):
            self._baselayer = nn.Sequential(*list(model.children())[:-2])
            inchannels = OUTPUT_DIM[self.backbone]
        if pool=='spoc':
            self.pool=SPoC()
        elif pool=='scale_avg':
            self.pool = scale_avg_pool()
        elif pool=='vlad':
            self.pool=VLAD()
        self.norm = L2N()
        self.usa_pca = use_pca
        self._linearlayer = nn.Linear(inchannels, numclasses)
        self.Flatten = Flatten()
        self.features = OrderedDict()

    def forward(self, x):
        for name, module in self._baselayer._modules.items():
            x = module(x)
            self.features[name] = x

        x = self.pool(x)
        self.features['pool'] = x
        x = self.norm(x)
        self.features['norm'] = x
        x = self.Flatten(x)

        #pca = PCA(whiten=True)
        #x=pca.fit_transform(tonumpy(x))
        '''pca_matrix = pca.components_
        pca_mean = pca.mean_
        pca_vars = pca.explained_variance_
        x=ApplyPcaAndWhitening(x,pca_matrix,pca_mean,pca_vars,pca_dims=1)'''

        #x=self.norm(totensor(x))
        out=x

        return out

class retrievalNet(nn.Module):
    def __init__(self,modelname,finetune=False, pool='gem',embedding_layer=True,**kwargs):
        super(retrievalNet, self).__init__()
        inchannels = 1000
        model=networks_map[modelname](pretrained=True)

        if modelname.startswith('resnet') or modelname.startswith('vgg'):
            if finetune:
                self._baselayer = list(model.children())[0]
            else:
                self._baselayer= nn.Sequential(*list(model.children())[:-2])
            inchannels = OUTPUT_DIM[modelname]
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
        self.dropout=nn.Dropout(p=0.5)
        self.Flatten = Flatten()
        self.features=OrderedDict()

    def forward(self,x,need_feature=True):
        for name,module in self._baselayer._modules.items():
            x=module(x)
            self.features[name]=x

        x=self.pool(x)
        self.features['pool'] = x
        x=self.Flatten(x)
        x=self.dropout(x)
        x=self.embedding_linear(x)
        x=self.norm(x)

        return x


class SiameseNet(nn.Module):
    def __init__(self,model):
        super(SiameseNet,self).__init__()
        self._baselayer = model
        self.norm = L2N()
        self.features=[]
    def forward(self,x1,x2):
        x1 = self.norm(self._baselayer(x1))
        x2 = self.norm(self._baselayer(x2))

        return x1,x2


class TripletNet(nn.Module):
    def __init__(self,model):
        super(TripletNet,self).__init__()
        #self._baselayer=Net(model,featuredim)
        self._baselayer =model
        self.norm=L2N()

    def forward(self,x1,x2,x3):
        x1=self.norm(self._baselayer(x1))
        x2=self.norm(self._baselayer(x2))
        x3=self.norm(self._baselayer(x3))
        return x1,x2,x3

def getModel(modelName,num_classes,Gpu,modeltype,cuda_gpu=True,pretrained=True,balanced=False,**kargs):

    if modeltype=='classification':

        kargs['num_classes']=num_classes
        kargs['name']=modelName

        mymodel = Net(networks_map[modelName](pretrained=pretrained),**kargs)

    elif modeltype=='siamese':

        kargs['num_classes'] = num_classes
        kargs['name'] = modelName
        mymodel=SiameseNet(Net(networks_map[modelName](pretrained=pretrained),**kargs))

    elif modeltype=='triplet':

        kargs['num_classes'] = num_classes
        kargs['name'] = modelName
        mymodel = TripletNet(Net(networks_map[modelName](pretrained=pretrained),**kargs))

    elif modeltype=='landmark':
        kargs={'numkeypoints':num_classes}
        mymodel= networks_map[modelName](pretrained=pretrained,**kargs)

    elif modeltype=='onlinepair':

        kargs['num_classes'] = num_classes
        mymodel = networks_map[modelName](pretrained=pretrained, **kargs)

    elif modeltype=='extractor':

        kargs['name'] = modelName
        kargs['num_classes'] = num_classes

        mymodel = extractor(networks_map[modelName](pretrained=pretrained),  **kargs)
    elif modeltype=='retrieval':

        kargs['name'] = modelName
        kargs['num_classes'] = num_classes

        mymodel = retrievalNet(modelName, **kargs)
    elif modeltype=='multitrain':

        kargs['num_classes'] = num_classes
        kargs['name'] = modelName
        mymodel = multibranchNet(networks_map[modelName](pretrained=pretrained), **kargs)
    elif modeltype=='glem':
        kargs['num_classes'] = num_classes
        kargs['name'] = modelName
        mymodel=GLEMweightNet(networks_map[modelName](pretrained=pretrained), **kargs)
    elif modeltype=='glemextractor':
        kargs['num_classes'] = num_classes
        kargs['name'] = modelName
        mymodel = GLEMweightExtractor(networks_map[modelName](pretrained=pretrained), **kargs)

    if cuda_gpu:
        print(Gpu)
        if balanced:
            mymodel=DataParallelModel(mymodel,device_ids=Gpu).cuda()
        else:

            mymodel = torch.nn.DataParallel(mymodel, device_ids=Gpu).cuda()
    return mymodel

if __name__=='__main__':
    qloader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=['/home/shibaorong/modelTorch/test.jpg'], imsize=224),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    kargs = {}
    kargs['name'] = 'resnet50'
    kargs['num_classes'] = 11
    initmodel=networks_map['resnet50'](pretrained=True);
    re=multibranchNet(initmodel,**kargs)

    for input in qloader:
        out=re(input)
        print(out)
