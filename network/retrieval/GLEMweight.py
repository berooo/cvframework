import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial
from network.outputdim import OUTPUT_DIM
from collections import OrderedDict
from util.util import *
from graph.pooling import *
class Nonlocal(nn.Module):
  def __init__(self,inplanes):
    super(Nonlocal,self).__init__()
    self.inter_planes=int(inplanes/2)
    self.g=nn.Conv2d(inplanes,self.inter_planes,1,1,0)
    self.theta=nn.Conv2d(inplanes,self.inter_planes,1,1,0)
    self.phi=nn.Conv2d(inplanes,self.inter_planes,1,1,0)

    self.W=nn.Sequential(
      nn.Conv2d(self.inter_planes,inplanes,1,1,0),
      nn.BatchNorm2d(inplanes)
    )

    nn.init.constant(self.W[1].weight,1)
    nn.init.constant(self.W[1].bias,0)
    self.sigmoid=nn.Sigmoid()

  def forward(self,x):
    batch_size=x.size(0)

    g_x=self.g(x).view(batch_size,self.inter_planes,-1)
    g_x=g_x.permute(0,2,1)

    theta_x=self.theta(x).view(batch_size,self.inter_planes,-1)
    theta_x=theta_x.permute(0,2,1)

    phi_x=self.phi(x).view(batch_size,self.inter_planes,-1)

    f=torch.matmul(theta_x,phi_x)
    f_div_C=F.softmax(f,dim=-1)

    y=torch.matmul(f_div_C,g_x)

    y=y.permute(0,2,1).contiguous()
    y=y.view(batch_size,self.inter_planes,*x.size()[2:])
    W_y=self.W(y)
    z=W_y+x

    return z

class GlobalLocalEmbedding(nn.Module):
  def __init__(self,in_channel):
    super(GlobalLocalEmbedding,self).__init__()
    self.non_local=Nonlocal(in_channel)
    self.conv1=nn.Conv2d(in_channel,in_channel,3,1,1)
    self.bn1=nn.BatchNorm2d(in_channel)

    self.conv2=nn.Conv2d(in_channel,in_channel,3,1,1)
    self.bn2=nn.BatchNorm2d(in_channel)
    self.relu=nn.ReLU(inplace=True)

  def forward(self,x):
    x=self.non_local(x)
    x=self.relu(self.bn1(self.conv1(x)))
    y=self.relu(self.bn2(self.conv2(x)))
    return y

class GLEMweightNet(nn.Module):
  def __init__(self, model, **kwargs):
    super(GLEMweightNet, self).__init__()
    self.backbone = kwargs['name']
    numclasses = kwargs['num_classes']
    inchannels = 1000
    if self.backbone.startswith('resnet'):
      self._baselayer = nn.Sequential(*list(model.children())[:-1])
      inchannels = OUTPUT_DIM[self.backbone]
    self.features = OrderedDict()
    self.pool=GlobalLocalEmbedding(OUTPUT_DIM[self.backbone])
    self._linearlayer = nn.Linear(inchannels, numclasses)
    self.Flatten = Flatten()

  def forward(self, x, need_feature=True):

    for name, module in self._baselayer._modules.items():
      x = module(x)
      self.features[name] = x

    x=self.pool(x)
    out = self.Flatten(x)
    out = self._linearlayer(out)

    return out

class GLEMweightExtractor(nn.Module):
  def __init__(self, model, **kwargs):
    super(GLEMweightExtractor, self).__init__()
    self.backbone = kwargs['name']
    numclasses = kwargs['num_classes']
    inchannels = 1000
    if self.backbone.startswith('resnet'):
      self._baselayer = nn.Sequential(*list(model.children())[:-1])
      inchannels = OUTPUT_DIM[self.backbone]
    self.features = OrderedDict()
    self.pool=GlobalLocalEmbedding(OUTPUT_DIM[self.backbone])
    self.extract_pool = SPoC()
    self.norm = L2N()
    self._linearlayer = nn.Linear(inchannels, numclasses)
    self.Flatten = Flatten()

  def forward(self, x, need_feature=True):

    for name, module in self._baselayer._modules.items():
      x = module(x)
      self.features[name] = x

    x=self.pool(x)
    x=self.extract_pool(x)
    out=self.norm(x)
    out = self.Flatten(out)

    return out