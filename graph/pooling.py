import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import graph.function as LF
from graph.normalization import L2N
from sklearn.cluster import KMeans
import numpy as np
# --------------------------------------
# Pooling layers
# --------------------------------------
from util.array_tool import totensor, tonumpy


class MAC(nn.Module):

  def __init__(self):
    super(MAC, self).__init__()

  def forward(self, x):
    return LF.mac(x)

  def __repr__(self):
    return self.__class__.__name__ + '()'


class SPoC(nn.Module):

  def __init__(self):
    super(SPoC, self).__init__()

  def forward(self, x):
    return LF.spoc(x)

  def __repr__(self):
    return self.__class__.__name__ + '()'


class GeM(nn.Module):

  def __init__(self, p=3, eps=1e-6):
    super(GeM, self).__init__()
    self.p = Parameter(torch.ones(1) * p)
    self.eps = eps

  def forward(self, x):
    return LF.gem(x, p=self.p, eps=self.eps)

  def __repr__(self):
    return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
      self.eps) + ')'


class GeMmp(nn.Module):

  def __init__(self, p=3, mp=1, eps=1e-6):
    super(GeMmp, self).__init__()
    self.p = Parameter(torch.ones(mp) * p)
    self.mp = mp
    self.eps = eps

  def forward(self, x):
    return LF.gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps)

  def __repr__(self):
    return self.__class__.__name__ + '(' + 'p=' + '[{}]'.format(self.mp) + ', ' + 'eps=' + str(self.eps) + ')'


class RMAC(nn.Module):

  def __init__(self, L=3, eps=1e-6):
    super(RMAC, self).__init__()
    self.L = L
    self.eps = eps

  def forward(self, x):
    return LF.rmac(x, L=self.L, eps=self.eps)

  def __repr__(self):
    return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class Rpool(nn.Module):

  def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
    super(Rpool, self).__init__()
    self.rpool = rpool
    self.L = L
    self.whiten = whiten
    self.norm = L2N()
    self.eps = eps

  def forward(self, x, aggregate=True):
    # features -> roipool
    o = LF.roipool(x, self.rpool, self.L, self.eps)  # size: #im, #reg, D, 1, 1

    # concatenate regions from all images in the batch
    s = o.size()
    o = o.view(s[0] * s[1], s[2], s[3], s[4])  # size: #im x #reg, D, 1, 1

    # rvecs -> norm
    o = self.norm(o)

    # rvecs -> whiten -> norm
    if self.whiten is not None:
      o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))

    # reshape back to regions per image
    o = o.view(s[0], s[1], s[2], s[3], s[4])  # size: #im, #reg, D, 1, 1

    # aggregate regions into a single global vector per image
    if aggregate:
      # rvecs -> sumpool -> norm
      o = self.norm(o.sum(1, keepdim=False))  # size: #im, D, 1, 1

    return o

  def __repr__(self):
    return super(Rpool, self).__repr__() + '(' + 'L=' + '{}'.format(self.L) + ')'


class scale_sum_pool(nn.Module):
  def __init__(self, p=[1, 2, 3]):
    super(scale_sum_pool, self).__init__()
    self.p = p

  def forward(self, x):
    n, c, h, w = x.size()
    out = []
    for i in self.p:
      o = LF.spoc(x, p=i).view(n, c, -1)
      out.append(o)

    res = torch.cat(out, axis=-1)
    return res

class scale_avg_pool(nn.Module):
  def __init__(self):
    super(scale_avg_pool,self).__init__()
    self.pool1=nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    self.pool2=nn.AvgPool2d(kernel_size=5,stride=1,padding=2)
    self.pool3=nn.AvgPool2d(kernel_size=7,stride=1,padding=3)

  def forward(self,x):
    p1=self.pool1(x)
    p2=self.pool2(x)
    p3=self.pool3(x)
    s1=LF.spoc(p1)
    s2=LF.spoc(p2)
    s3=LF.spoc(p3)
    s=s1+s2+s3
    #res=torch.cat((s1,s2),dim=1)
    return s

class weighted_pool(nn.Module):
  def __init__(self,p=2048):
    super(weighted_pool, self).__init__()
    self.p =Parameter(torch.rand([1,p,1,1]))
  def forward(self,x):
    y=x*self.p
    y = y.view(-1, x.size(1), x.size(2) * x.size(3))  # b x c x hw
    return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)


class VLAD(nn.Module):
  def __init__(self,k=20):
    super(VLAD,self).__init__()
    self.k=50
    self.model=KMeans(n_clusters=k)
  def forward(self,x):
    y = x.view(-1, x.size(1), x.size(2) * x.size(3))  # b x c x hw
    vladres=np.zeros([y.size(0),self.k,y.size(-1)])
    for i in range(y.size(0)):
      feature=tonumpy(y[i])
      c,hw=feature.shape
      self.model.fit(feature)
      clusters=self.model.cluster_centers_
      labels=self.model.labels_
      for j in range(c):
          label=labels[j]
          vladres[i,label,:]+=feature[j]-clusters[label]
    vladres=totensor(vladres)
    return vladres


