
import pickle
from util.autoaugment import ImageNetPolicy
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import json
import requests
import random as rd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from datasets.imageListDateset import ImagesFromList


class OnlineTripletData(Dataset):
  def __init__(self,path, autoaugment,outputdim, imsize=224, qsize=2000,transform=transforms.Compose([transforms.ToTensor()])):
    '''
    :param path: json文件路径
    :param qsize: 每个epoch需要构建多少个三元组
    :param transform:
    '''
    self.data=json.load(open(path))
    self.autoaugment = autoaugment
    self.transform = transform
    self.imsize = imsize
    self.qsize=qsize
    self.outputdim=outputdim
    self.print_freq=10
    self.ppool = {}
    self.clusters = []

    for d in self.data:
      if d['label_id'] not in self.clusters:
        self.clusters.append(d['label_id'])
        self.ppool[str(d['label_id'])] = [d['filenames']]
      else:
        self.ppool[str(d['label_id'])].append(d['filenames'])

    self.searchsope = int(len(self.data) // len(self.clusters))

  def create_epoch_tuples(self, net):
    '''
    三元组挖掘策略：
    搜索范围search scope=数据集的数目//类的数目
    1、将所有数据通过网络提取特征，特征之间矩阵乘积，每个元素(i,j)对应这第i个和第j个之间的相似性；然后dim=0进行sort。
    2、随机从所有数据中选出k个作为查询数据，遍历每条查询数据，
    -对每个查询数据，在其对应的类中随机选择一条数据作为positive；
    -对每个查询数据，取出sort之后的矩阵的对应那一列，遍历那一列的每一个元素(元素的含义是Index)，如果该index的数据类和查询数据相同，那么continue;否则，当作negative，产生一个triplet对，加入result中。每次遍历一列，产生len(clusters)-1个negative，产生len(clusters)-1个triplet对；
    3、最后对于所有产生的triplet对，随机采样k个triplet对，作为模型输入item，并返回
    '''
    self.qpool = []

    net.cuda()
    #net.eval()

    with torch.no_grad():
      print('>> Extracting descriptors for query images...')
      loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=[i['filenames'] for i in self.data], imsize=self.imsize,
                       transform=self.transform),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
      )

      poolvecs = torch.zeros(self.outputdim, len(self.data)).cuda()
      for i,input in enumerate(loader):
        out,_=net(input.cuda())
        poolvecs[:, i] = out.data.squeeze()
        if (i + 1) % self.print_freq == 0 or (i + 1) == self.qsize:
          print('\r>>>> {}/{} done...'.format(i + 1, self.qsize), end='')
      print('')

    scores = torch.mm(poolvecs.t(), poolvecs)
    scores, ranks = torch.sort(scores, dim=0, descending=True)

    queryids=torch.randperm(len(self.data))[:self.qsize]

    for id in queryids:
      label=self.data[id]['label_id']
      anchor=self.data[id]['filenames']
      repeat=-1

      while repeat==-1:
        positive=rd.sample(self.ppool[str(label)],1)[0]
        if positive!=anchor:
          repeat+=1
      count=len(self.clusters)-1

      for j in range(self.searchsope):
        nid=ranks[id,j]
        if self.data[nid]['label_id']==label:
          continue
        negative=self.data[nid]['filenames']
        nlabel=self.data[nid]['label_id']
        self.qpool.append((anchor,positive,negative,label,nlabel))
        count-=1
        if count<0:
          break

    rd.shuffle(self.qpool)

  def __getitem__(self, index):
    if index>=len(self.qpool):
      return None,None,None

    imgpath1, imgpath2, imgpath3,plabel,nlabel=self.qpool[index]
    try:
      img1 = Image.open(imgpath1).convert('RGB')
      img2 = Image.open(imgpath2).convert('RGB')
      img3 = Image.open(imgpath3).convert('RGB')
    except:
      return self.__getitem__(index + 1)

    if self.autoaugment:
      policy = ImageNetPolicy()
      img1 = policy(img1)
      img2 = policy(img2)
      img3=policy(img3)

    img1 = img1.resize((self.imsize, self.imsize))
    img1 = np.array(img1)
    img1 = self.transform(img1)

    img2 = img2.resize((self.imsize, self.imsize))
    img2 = np.array(img2)
    img2 = self.transform(img2)

    img3 = img3.resize((self.imsize, self.imsize))
    img3 = np.array(img3)
    img3 = self.transform(img3)

    out=[]
    out.append(img1)
    out.append(img2)
    out.append(img3)

    return out,plabel,nlabel

  def __len__(self):
    return self.qsize