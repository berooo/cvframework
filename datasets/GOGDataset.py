import pickle
import torch

from datasets.imageListDateset import ImagesFromList
from util.autoaugment import ImageNetPolicy
import numpy as np
import torchvision.transforms as transforms
import json
from PIL import Image
from torch.utils.data import Dataset
from network import OUTPUT_DIM
from util.util import loadquery
import random
import csv
from util.array_tool import tonumpy

root='/mnt/sdb/shibaorong/data/googleLandmark/train'

class GotripletDataset(Dataset):
  def __init__(self,path,autoaugment,height=224,width=224,transform=transforms.Compose([transforms.ToTensor()])):
    labels = []
    reader=csv.reader(open(path,'r'))
    result={}
    for index,item in enumerate(reader):
        if reader.line_num == 1:
            continue
        label=item[1]
        f=item[0]
        fpath = root + '/' + f[0] + '/' + f[1] + '/' + f[2] + '/' + f + '.jpg'
        if label in result:
          result[label].append(fpath)
        else:
          result[label]=[fpath]

        labels.append(label)

    self.labels = labels
    self.datapool=result
    self.autoaugment = autoaugment
    self.transform = transform
    self.height = height
    self.width = width
    self.K=512
    self.mining_batch_size=2048
    self.T=30
    self.triplets=[]


  def __getitem__(self, index):
    imgpath1, imgpath2, imgpath3 = self.triplets[index]
    # label1, label2, label3 = self.labels[index]
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
      img3 = policy(img3)

    img1 = img1.resize((self.width, self.height))
    img1 = np.array(img1)
    img1 = self.transform(img1)

    img2 = img2.resize((self.width, self.height))
    img2 = np.array(img2)
    img2 = self.transform(img2)

    img3 = img3.resize((self.width, self.height))
    img3 = np.array(img3)
    img3 = self.transform(img3)

    return img1, img2, img3

  def create_triplet(self,mymodel,featuredim=2048):
    choose_labels=random.sample(self.labels,self.K)
    candidates=[]
    for label in choose_labels:
      for name in self.datapool[label]:
        candidates.append([name,label])
    #candidates=random.sample(candidates,self.mining_batch_size)

    random.shuffle(candidates)
    #vecs=torch.zeros([featuredim,self.mining_batch_size])
    vecs = torch.zeros([featuredim, len(candidates)])
    tripletlist=[]

    mymodel.eval()
    with torch.no_grad():
      print('>> Extracting descriptors for query images...')
      qloader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=[i[0] for i in candidates], imsize=self.height,
                       transform=self.transform),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
      )
      for i, input in enumerate(qloader):
        out = mymodel(input.cuda())
        vecs[:, i] = out
        if (i + 1) % 10 == 0:
          print('\r>>>> {}/{} done...'.format(i + 1, len(candidates)), end='')

    qindexs = random.sample([i for i in range(len(candidates))], self.mining_batch_size)
    qvecs = vecs[:, qindexs]
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)

    for i in range(ranks.shape[1]):

      qlabel=candidates[qindexs[i]][1]
      aimg=candidates[qindexs[i]][0]
      pimg=None
      nimg=None
      pmark=0
      nmark=0
      rank=ranks[:,i]
      mark = False
      for indj in range(len(candidates)):
        if indj>self.T:
          continue
        j=rank[indj]
        rlabel=candidates[j][1]

        if qlabel==rlabel:
          if mark:
            pimg = candidates[j][0]
            nimg=candidates[rank[indj-1]][0]
            tripletlist.append((aimg, pimg, nimg))
          mark=False
        else:
          mark=True
        '''if (rlabel!=qlabel) and (nimg is None):
          nimg=candidates[j][0]
          nmark=indj
          if indj>2:
            pimg=candidates[rank[indj-1]][0]
            pmark=indj

        elif (rlabel==qlabel) and (nimg is not None):
          pimg=candidates[j][0]
          pmark=indj
        if pimg is not None and nimg is not None and pmark-nmark<self.T:
          tripletlist.append((aimg,pimg,nimg))
          break'''


    self.triplets=tripletlist
    '''if len(self.triplets)==0:
      self.create_triplet(mymodel,featuredim)
      if(self.T<50):
        self.T+=10
      else:
        self.K+=10'''


  def __len__(self):
    return len(self.triplets)


class GoclassDataset(Dataset):
  def __init__(self,path,autoaugment,height=224,width=224,transform=transforms.Compose([transforms.ToTensor()])):
    filenames,labels = [],[]
    reader=csv.reader(open(path,'r'))
    labelnum={}
    for index,item in enumerate(reader):
        if reader.line_num == 1:
            continue
        label=item[1]
        f=item[0]
        fpath = root + '/' + f[0] + '/' + f[1] + '/' + f[2] + '/' + f + '.jpg'
        filenames.append(fpath)

        if label not in labelnum:
            l=len(labelnum)
            labels.append(l)
            labelnum[label]=l
        else:
            labels.append(labelnum[label])

    self.labels = labels
    self.filenames=filenames
    self.autoaugment = autoaugment
    self.transform = transform
    self.height = height
    self.width = width

  def __getitem__(self, index):
      imgpath = self.filenames[index]
      try:
          img = Image.open(imgpath).convert('RGB')
      except:
          return self.__getitem__(index + 1)

      if self.autoaugment:
          policy = ImageNetPolicy()
          img = policy(img)

      img = img.resize((self.width, self.height))
      img = np.array(img)
      img = self.transform(img)
      label = self.labels[index]
      return img, label

  def __len__(self):
      return len(self.filenames)