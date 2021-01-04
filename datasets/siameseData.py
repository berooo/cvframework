from util.autoaugment import ImageNetPolicy
import numpy as np
import torchvision.transforms as transforms
import json
from PIL import Image
from torch.utils.data import Dataset

class SiameseData(Dataset):
  def __init__(self,path,autoaugment,height=224,width=224,transform=transforms.Compose([transforms.ToTensor()])):
    filenames,label=[],[]
    data=json.load(open(path))
    for d in data:
      f=d[0]['filenames'],d[1]['filenames']
      l=d[0]['label_id'],d[1]['label_id']
      filenames.append(f)
      label.append(l)

    self.filenames=filenames
    self.labels=label
    self.autoaugment=autoaugment
    self.transform=transform
    self.height=height
    self.width=width

  def __getitem__(self, index):
    imgpath1,imgpath2 = self.filenames[index]
    try:
      img1=Image.open(imgpath1).convert('RGB')
      img2 = Image.open(imgpath2).convert('RGB')
    except:
      return self.__getitem__(index+1)

    if self.autoaugment:
      policy=ImageNetPolicy()
      img1=policy(img1)
      img2=policy(img2)

    img1=img1.resize((self.width,self.height))
    img1 = np.array(img1)
    img1=self.transform(img1)

    img2 = img2.resize((self.width, self.height))
    img2 = np.array(img2)
    img2 = self.transform(img2)

    label1,label2=self.labels[index]
    return img1,label1,img2,label2

  def __len__(self):
    return len(self.filenames)

class SiameseIDData(Dataset):
  def __init__(self,path,autoaugment,height=224,width=224,transform=transforms.Compose([transforms.ToTensor()])):
    filenames,label,ids=[],[],[]
    data=json.load(open(path))
    for d in data:
      f=d[0]['filenames'],d[1]['filenames']
      l=d[0]['label_id'],d[1]['label_id']
      id=d[0]['ID'],d[1]['ID']
      filenames.append(f)
      label.append(l)
      ids.append(id)

    self.filenames=filenames
    self.labels=label
    self.autoaugment=autoaugment
    self.transform=transform
    self.height=height
    self.width=width
    self.ids=ids

  def __getitem__(self, index):
    imgpath1,imgpath2 = self.filenames[index]
    try:
      img1=Image.open(imgpath1).convert('RGB')
      img2 = Image.open(imgpath2).convert('RGB')
    except:
      return self.__getitem__(index+1)

    if self.autoaugment:
      policy=ImageNetPolicy()
      img1=policy(img1)
      img2=policy(img2)

    img1=img1.resize((self.width,self.height))
    img1 = np.array(img1)
    img1=self.transform(img1)

    img2 = img2.resize((self.width, self.height))
    img2 = np.array(img2)
    img2 = self.transform(img2)

    label1,label2=self.labels[index]
    id1,id2=self.ids[index]
    return img1,label1,id1,img2,label2,id2

  def __len__(self):
    return len(self.filenames)