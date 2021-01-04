
import pickle
from random import random

import torch
import torchvision
import os
from datasets.imageListDateset import ImagesFromList
from util.autoaugment import ImageNetPolicy
import numpy as np
import torchvision.transforms as transforms
import json
from PIL import Image
from torch.utils.data import Dataset
from network import OUTPUT_DIM
from util.util import loadquery
from datasets.util import eyetransfrom


class FolderDataset(Dataset):
  def __init__(self,root,mode='train'):
    filenames, labels = [], []
    self.mode=mode

    if mode=='train':
      for dirpath, dirname, fname in os.walk(root):
        for f in fname:
          if f.endswith('txt'):
            continue
          filename = os.path.join(dirpath, f)
          labelname = int(os.path.basename(dirpath).split('_')[-1])
          filenames.append(filename)
          labels.append(labelname)
      self.filenames = filenames
      self.labels = labels
      self.transfrom=transforms.Compose([
      ImageNetPolicy(),
      transforms.RandomResizedCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    else:
      for dirpath, dirname, fname in os.walk(root):
        for f in fname:
          if f.endswith('txt'):
            continue
          filename = os.path.join(dirpath, f)
          filenames.append(filename)

      self.filenames = filenames
      self.transfrom=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

  def __getitem__(self, index):
    imgpath = self.filenames[index]
    try:
      img = Image.open(imgpath).convert('RGB')
    except:
      return self.__getitem__(index + 1)


    img=self.transfrom(img)

    if self.mode=='train':
      label = self.labels[index]
      return img,label
    else:
      return img,imgpath

  def __len__(self):
    return len(self.filenames)

class myDataset(Dataset):
  def __init__(self,path,autoaugment,height=224,width=224,transform=transforms.Compose([transforms.ToTensor()])):
    filenames, labels = [], []
    data = json.load(open(path))
    for d in data:
      filenames.append(d['filenames'])
      labels.append(d['label_id'])
    self.filenames=filenames
    self.labels=labels
    self.autoaugment=autoaugment
    self.transform=transform
    self.height=height
    self.width=width

  def __getitem__(self, index):
    imgpath=self.filenames[index]
    try:
      img=Image.open(imgpath).convert('RGB')
    except:
      return self.__getitem__(index+1)

    if self.autoaugment:
      policy=ImageNetPolicy()
      img=policy(img)

    img=img.resize((self.width,self.height))
    img = np.array(img)
    img=self.transform(img)
    label=self.labels[index]
    return img,label,imgpath

  def __len__(self):
    return len(self.filenames)


class IDDataset(Dataset):
  def __init__(self,path,autoaugment,height=224,width=224,transform=transforms.Compose([transforms.ToTensor()])):
    filenames, ids,labels = [], [],[]
    data = json.load(open(path))
    for d in data:
      filenames.append(d['filenames'])
      ids.append(d['ID'])
      labels.append(d['label_id'])
    self.filenames=filenames
    self.ids=ids
    self.autoaugment=autoaugment
    self.transform=transform
    self.height=height
    self.width=width
    self.labels = labels

  def __getitem__(self, index):
    imgpath=self.filenames[index]
    try:
      img=Image.open(imgpath).convert('RGB')
    except:
      return self.__getitem__(index+1)

    if self.autoaugment:
      policy=ImageNetPolicy()
      img=policy(img)

    img=img.resize((self.width,self.height))
    img = np.array(img)
    img=self.transform(img)
    id=self.ids[index]
    label = self.labels[index]
    return img,label,imgpath,id

  def __len__(self):
    return len(self.filenames)


class IDDatasetWithoutimg(Dataset):
  def __init__(self,path):
    filenames, ids,labels = [], [],[]
    data = json.load(open(path))
    for d in data:
      filenames.append(d['filenames'])
      ids.append(d['ID'])
      labels.append(d['label_id'])
    self.filenames=filenames
    self.ids=ids
    self.labels = labels

  def __getitem__(self, index):
    imgpath=self.filenames[index]
    id=self.ids[index]
    label = self.labels[index]
    return label,imgpath,id

  def __len__(self):
    return len(self.filenames)


class balancedDataset(Dataset):

  def __init__(self,path,imsize=224):
    filenames, labels = [], []
    data = json.load(open(path))
    sample_pool={}
    self.totallen=len(data)
    i=0
    for d in data:
      filename=d['filenames']
      label=d['label_id']
      if label not in sample_pool:
        sample_pool[label]=[filename]
        i+=1
      else:
        sample_pool[label].append(filename)
    self.batchnum=int(self.totallen/i)
    self.pool=sample_pool
    self.filenames = filenames
    self.labels = labels
    self.transform = eyetransfrom(imsize)
    self.cadidates=[]

  def balanced_sample(self):

    for i in range(self.batchnum):
      for j in self.pool:
        p=self.pool[j]
        imgpath=random.sample(p,1)[0]
        self.cadidates.append([imgpath,j])
    random.shuffle(self.cadidates)

  def __getitem__(self, item):
    imgpath,label=self.cadidates[item]
    try:
      img = Image.open(imgpath).convert('RGB')
    except Exception as e:
      print(e)
      return self.__getitem__(item+1)
    finally:
      img = self.transform(img)

    return img,label


  def __len__(self):
    return len(self.cadidates)