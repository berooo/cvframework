import torch

from datasets.imageListDateset import ImagesFromList
from util.autoaugment import ImageNetPolicy
import numpy as np
import torchvision.transforms as transforms
import json
from PIL import Image
from torch.utils.data import Dataset
from datasets.commonDataset import IDDatasetWithoutimg
from util.util import loadquery
from network import OUTPUT_DIM
from torch.autograd import Variable
from util.array_tool import scalar
import random

class TripletData(Dataset):
  def __init__(self, path, autoaugment, height=224, width=224, transform=transforms.Compose([transforms.ToTensor()])):
    filenames, label = [],[]
    data = json.load(open(path))
    for d in data:
      f = d[0]['filenames'], d[1]['filenames'],d[2]['filenames']
      l = d[0]['label_id'], d[1]['label_id'],d[2]['label_id']
      filenames.append(f)
      label.append(l)

    self.filenames = filenames
    self.labels = label
    self.autoaugment = autoaugment
    self.transform = transform
    self.height = height
    self.width = width

  def __getitem__(self, index):
    imgpath1, imgpath2,imgpath3 = self.filenames[index]
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

  def __len__(self):
    return len(self.filenames)


class tripletAndPathData:
  def __init__(self, path, autoaugment, height=224, width=224, transform=transforms.Compose([transforms.ToTensor()])):
    filenames, label = [],[]
    data = json.load(open(path))
    for d in data:
      f = d[0]['filenames'], d[1]['filenames'],d[2]['filenames']
      l = d[0]['label_id'], d[1]['label_id'],d[2]['label_id']
      filenames.append(f)
      label.append(l)

    self.filenames = filenames
    self.labels = label
    self.autoaugment = autoaugment
    self.transform = transform
    self.height = height
    self.width = width

  def __getitem__(self, index):
    imgpath1, imgpath2,imgpath3 = self.filenames[index]
    label1,label2,label3=self.labels[index]
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

    img1 = img1.resize((self.width, self.height))
    img1 = np.array(img1)
    img1 = self.transform(img1)

    img2 = img2.resize((self.width, self.height))
    img2 = np.array(img2)
    img2 = self.transform(img2)

    img3 = img3.resize((self.width, self.height))
    img3 = np.array(img3)
    img3 = self.transform(img3)

    return img1,label1, img2,label2, img3,label3

  def __len__(self):
    return len(self.filenames)

class miningTripletData():
  def __init__(self,args,transform=transforms.Compose([transforms.ToTensor()])):
    self.path=args.jsonfile
    self.autoaugment=args.autoaugment
    self.triplet_pool=[]
    filenames, ids, labels = [], [], []
    records={}
    data = json.load(open(self.path))
    for d in data:
      filenames.append(d['filenames'])
      ids.append(d['ID'])
      labels.append(d['label_id'])
      if d['label_id'] not in records:
        records[d['label_id']]=1
      else:
        records[d['label_id']] += 1
    self.filenames = filenames
    self.ids = ids
    self.labels = labels
    self.height=args.height
    self.width=args.width
    self.transform = transform
    self.qsize=50
    self.qscale=100
    for d in records:
      self.qscale=min(self.qscale,records[d])

  def create_triplet_classbased_1(self, mymodel, args):
    print('create_triplet_classbased_1....................')

    vecs = torch.zeros([OUTPUT_DIM[args.backbone], len(self.filenames)])
    tripletlist = []
    #candidatesindexs = sorted(random.sample([i for i in range(len(self.filenames))], int(len(self.filenames) / args.classnum)))
    candidatesindexs = self.sample(args.classnum)
    mymodel.eval()
    with torch.no_grad():
      print('>> Extracting descriptors for query images...')
      qloader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=self.filenames, imsize=self.height,
                       transform=self.transform),
        batch_size=1, shuffle=False, num_workers=80, pin_memory=True
      )
      for i, input in enumerate(qloader):
        out = mymodel(input.cuda())
        if isinstance(out, dict):
          out = out['feature']
        vecs[:, i] = out
        if (i + 1) % 10 == 0:
          print('\r>>>> {}/{} done...'.format(i + 1, len(self.filenames)), end='')

    #candidatesindexs = random.sample([i for i in range(len(self.filenames))], int(len(self.filenames) / args.classnum))
    qvecs = vecs[:, candidatesindexs]
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)

    for i in range(len(candidatesindexs)):

      qlabel = self.labels[candidatesindexs[i]]
      aimg = self.filenames[candidatesindexs[i]]
      pimg = None
      nimg = None

      rank = ranks[:, i]
      for indj in range(len(self.filenames)):
        if indj>self.qscale:
          continue
        j = rank[indj]
        rlabel = self.labels[j]
        if (rlabel != qlabel) and (nimg is None):
          nimg = self.filenames[j]
          nlabel=self.labels[j]

          if indj > 2:
            pimg = self.filenames[rank[indj - 1]]
            plabel=self.labels[rank[indj - 1]]

        elif (rlabel == qlabel) and (nimg is not None):
          pimg = self.filenames[j]
          plabel=self.labels[j]

        if pimg is not None and nimg is not None:
          tripletlist.append([(aimg,qlabel), (pimg,plabel), (nimg,nlabel)])

    self.triplet_pool = tripletlist

  def sample(self,classnum):
    numavg=int(len(self.filenames)/(classnum*classnum*classnum))
    classes=[str(i) for i in range(classnum)]
    nums=[numavg]*classnum
    candidatesindexes=[]
    numdict=dict(zip(classes,nums))
    count=0
    while True:

      index=random.choice([i for i in range(len(self.filenames))])
      if numdict[str(self.labels[index])]>0 and (index not in candidatesindexes):
        candidatesindexes.append(index)
        numdict[str(self.labels[index])]-=1
        print('{},{}'.format(str(self.labels[index]),numdict[str(self.labels[index])]))
        if numdict[str(self.labels[index])]==0:
          count+=1
      if count==classnum:
        break
    return candidatesindexes

  def create_triplet_classbased_2(self,mymodel, args):
    print('create_triplet_classed_based_2....................')

    #candidatesindexs = random.sample([i for i in range(len(self.filenames))], int(len(self.filenames) / args.classnum))
    candidatesindexs=self.sample(args.classnum)

    vecs = torch.zeros([OUTPUT_DIM[args.backbone], len(self.filenames)])
    tripletlist = []

    mymodel.eval()
    with torch.no_grad():
      print('>> Extracting descriptors for query images...')
      qloader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=self.filenames, imsize=self.height,
                       transform=self.transform),
        batch_size=1, shuffle=False, num_workers=80, pin_memory=True
      )
      for i, input in enumerate(qloader):
        out = mymodel(input.cuda())
        if isinstance(out,dict):
          out=out['feature']
        vecs[:, i] = out
        if (i + 1) % 10 == 0:
          print('\r>>>> {}/{} done...'.format(i + 1, len(self.filenames)), end='')

    qvecs= vecs[:, candidatesindexs]
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)

    for i in range(len(candidatesindexs)):

      qlabel = self.labels[candidatesindexs[i]]
      aimg = self.filenames[candidatesindexs[i]]

      rank = ranks[:, i]
      mark=False
      for indj in range(len(self.filenames)):
        if indj > self.qscale:
          continue
        j = rank[indj]
        rlabel = self.labels[j]

        if qlabel == rlabel:
          if mark:
            pimg = self.filenames[j]
            nimg = self.filenames[rank[indj - 1]]
            nlabel=self.labels[rank[indj-1]]
            tripletlist.append([(aimg,rlabel), (pimg,rlabel), (nimg,nlabel)])
          mark = False

        else:
          mark = True

      self.triplet_pool = tripletlist

  def create_triplet_clusterbased(self, mymodel, ClusterInfo, args):
    print('create_triplet....................')
    iddict, _, _ = ClusterInfo
    gnd = loadquery(args.valdata_dir)
    mymodel.eval()
    with torch.no_grad():

      featurelist = []
      #pathlist = []
      clusterlabelist = []
      #classlabelist = []

      for i in range(len(self.filenames)):
        ID = str(self.ids[i])
        feature = iddict[ID]['feature']
        featurelist.append(feature)
        clusterlabel = iddict[ID]['label']
        clusterlabelist.append(clusterlabel)

      qindexs=random.sample([i for i in range(len(featurelist))],self.qsize)
      #qindexs = np.arange(len(self.filenames))[np.in1d(self.filenames, [i['queryimgid'] for i in gnd])]
      '''newgnd = [self.filenames[i] for i in qindexs]
      g = [[i['queryimgid'] for i in gnd].index(j) for j in newgnd]
      gnd = [gnd[i] for i in g]'''
      vecs = np.transpose(np.array(featurelist))
      qvecs = vecs[:, qindexs]
      scores = np.dot(vecs.T, qvecs)
      ranks = np.argsort(-scores, axis=0)

      triplet_pool = []
      for i in range(ranks.shape[1]):
        rank = ranks[:, i]
        aimg = self.filenames[qindexs[i]]
        aclusterlabel = clusterlabelist[qindexs[i]]
        aclasslabel = self.labels[qindexs[i]]
        mark = False
        '''p=[self.filenames[g] for g in rank]
        po=gnd[i]['ok']
        pos = np.arange(len(p))[np.in1d(p, po)]'''
        for j in range(rank.shape[0]):
          if j>=self.qscale:
            continue
          #jmark=pos[j]
          rindex = rank[j]
          jpath = self.filenames[rindex]
          jclusterlabel = clusterlabelist[rindex]
          jclasslabel = self.labels[rindex]

          if jclasslabel == aclasslabel and jclusterlabel == aclusterlabel:
            if mark:
              pimg = jpath
              nimg = self.filenames[rank[j - 1]]
              triplet_pool.append([(aimg,aclasslabel), (pimg,jclasslabel), (nimg,self.labels[rank[j-1]])])

            mark = False

          else:
            mark=True
          '''elif jclasslabel!=aclasslabel and jclusterlabel!=aclusterlabel:
                      mark = True'''

    self.triplet_pool = triplet_pool

  def __getitem__(self, index):
    (imgpath1,label1), (imgpath2,label2), (imgpath3,label3) = self.triplet_pool[index]
    #label1, label2, label3 = self.labels[index]
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

    img1 = self.transform(img1)
    img2 = self.transform(img2)
    img3 = self.transform(img3)

    return (img1,label1), (img2,label2),  (img3,label3)


  def __len__(self):
    return len(self.triplet_pool)



    '''idlist.append(id[0])
        pathlist.append(path[0])

        if torch.cuda.is_available():
          img = img.cuda()
          label = label.cuda()

        img = img.float()
        #img = Variable(img)

        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = mymodel(img)
        poolvecs[:, index] = out
        if (index + 1) % 10 == 0:
          print('\r>>>> {}/{} done...'.format(index + 1, len(self.filenames)), end='')

      qindexs = np.arange(len(self.filenames))[np.in1d(pathlist, [i['queryimgid'] for i in gnd])]
      newgnd = [pathlist[i] for i in qindexs]
      g = [[i['queryimgid'] for i in gnd].index(j) for j in newgnd]
      gnd = [gnd[i] for i in g]

      vecs = poolvecs.cpu().numpy()
      qvecs = vecs[:, qindexs]

      # search, rank, and print
      scores = np.dot(vecs.T, qvecs)
      ranks = np.argsort(-scores, axis=0)

      for i in qindexs:
        feature=ClusterInfo'''
    '''loader = torch.utils.data.DataLoader(
           IDDatasetWithoutimg(path=self.path),
           batch_size=1, shuffle=False, num_workers=0, pin_memory=True
         )
         for index, ( classlabel, path, id) in enumerate(loader):'''