import torch.nn as nn
from collections import OrderedDict
from util.util import *

class dshNet(nn.Module):
  def __init__(self,pretrained=True,**kwargs):
    super(dshNet,self).__init__()
    featuredim=kwargs['num_classes']
    self.baselayer=nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=True)),
      ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
      ('bn1',nn.BatchNorm2d(32)),
      ('relu1', nn.ReLU(inplace=True)),
      ('conv2', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=True)),
      ('pool2', nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
      ('bn2',nn.BatchNorm2d(32)),
      ('relu2', nn.ReLU(inplace=True)),
      ('conv3', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=True)),
      ('bn3', nn.BatchNorm2d(32)),
      ('relu3', nn.ReLU(inplace=True)),
      ('pool3', nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
      ('flatten',Flatten()),
      ('ip1',nn.Linear(32*28*28,500)),
      ('relu4',nn.ReLU(inplace=True)),
      ('ip2',nn.Linear(500,featuredim))
    ]))
    self.features=OrderedDict()

  def forward(self,x):
    for name, module in self.baselayer._modules.items():
      x = module(x)
      self.features[name] = x

    return x

  def getfeatures(self):
    return self.features

