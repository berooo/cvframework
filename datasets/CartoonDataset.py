import os
import re
from PIL import Image
import cv2
import numpy as np
import random
import torchvision.transforms as transforms
import torch.utils.data

from datasets.imageListDateset import ImagesFromList
from util.autoaugment import ImageNetPolicy

class preclsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,imsize=224):
        label_dict={}

        for index,name in enumerate(os.listdir(data_path)):
            imgroot=os.path.join(data_path,name)
            label_dict[index]={'C':[],'P':[]}
            for imgname in os.listdir(imgroot):
                imgpath=os.path.join(imgroot,imgname)
                if imgname[0]=='C':
                    label_dict[index]['C'].append(imgpath)
                else:
                    label_dict[index]['P'].append(imgpath)

        self.innerdata=self.create_epoch(label_dict)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.imsize=imsize

    def create_epoch(self,label_dict):

        pairs=[]
        for key in label_dict:
            for p in label_dict[key]['C']:
                pairs.append((key,p))

        return pairs

    def __getitem__(self, index):
        label,pimgf=self.innerdata[index]
        policy=ImageNetPolicy()

        pimg=Image.open(pimgf).convert('RGB')
        pimg=policy(pimg)
        pimg = pimg.resize((self.imsize, self.imsize))
        pimg = np.array(pimg)
        pimg = self.transform(pimg)

        return pimg,label

    def __len__(self):
        return len(self.innerdata)

class CartoonDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,imsize=224):

        C = []
        P = []

        for index, name in enumerate(os.listdir(data_path)):
            imgroot = os.path.join(data_path, name)
            for imgname in os.listdir(imgroot):
                imgpath = os.path.join(imgroot, imgname)
                if imgname[0] == 'C':
                    C.append((index, imgpath))
                else:
                    P.append((index, imgpath))

        self.c = C
        self.p = P
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.imsize=imsize
        self.num_epoch=2000

    def create_epoch(self):
        pool=[]
        positive=0
        negative=0
        while True:

            if positive>=2000 and negative>=2000:
                break
            c=random.sample(self.c,1)[0]
            p=random.sample(self.p,1)[0]
            if c[0]==p[0]:
                if positive>=2000:
                    continue
                pool.append((c[1],p[1],c[0],p[0],1))
                positive+=1

            else:
                if negative>=2000:
                    continue
                pool.append((c[1], p[1],c[0],p[0], 0))
                negative+= 1
        random.shuffle(pool)
        self.pool=pool

    def __getitem__(self, index):
        imgpath1, imgpath2, label1,label2,target = self.pool[index]
        # label1, label2, label3 = self.labels[index]
        try:
            img1 = Image.open(imgpath1).convert('RGB')
            img2 = Image.open(imgpath2).convert('RGB')

        except:
            return self.__getitem__(index + 1)

        policy = ImageNetPolicy()

        img1 = policy(img1)
        img2 = policy(img2)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1,img2,label1,label2,target

    def __len__(self):
        return len(self.pool)

class CartoonDataset_tri(torch.utils.data.Dataset):
    def __init__(self, data_path,imsize=224):

        C=[]
        P=[]

        for index,name in enumerate(os.listdir(data_path)):
            imgroot=os.path.join(data_path,name)
            for imgname in os.listdir(imgroot):
                imgpath=os.path.join(imgroot,imgname)
                if imgname[0]=='C':
                    C.append((index,imgpath))
                else:
                    P.append((index,imgpath))

        self.c=C
        self.p=P
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.imsize=imsize
        self.qscale=30

    def create_epoch(self, branch_c, branch_p):
        querylen=int(len(self.c)*2/3)
        candidatesindexs=random.sample([i for i in range(len(self.c))], querylen)
        qvecs=torch.zeros([2048,querylen])
        vecs = torch.zeros([2048, len(self.p)])

        with torch.no_grad():
            print('>> Extracting descriptors for query images...')
            qloader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.c[i][1] for i in candidatesindexs], imsize=self.imsize,
                               transform=self.transform),
                batch_size=1, shuffle=False, num_workers=0, pin_memory=True
            )
            for i, input in enumerate(qloader):
                out = branch_c(input.cuda())
                if isinstance(out, dict):
                    out = out['feature']
                qvecs[:, i] = out
                if (i + 1) % 10 == 0:
                    print('\r>>>> {}/{} done...'.format(i + 1, querylen), end='')

            print('>> Extracting descriptors for gallery images...')
            qloader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[i[1] for i in self.p], imsize=self.imsize,
                               transform=self.transform),
                batch_size=1, shuffle=False, num_workers=0, pin_memory=True
            )
            for i, input in enumerate(qloader):
                out = branch_p(input.cuda())
                if isinstance(out, dict):
                    out = out['feature']
                vecs[:, i] = out
                if (i + 1) % 10 == 0:
                    print('\r>>>> {}/{} done...'.format(i + 1, len(self.p)), end='')

        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        tripletlist=[]

        for i in range(len(candidatesindexs)):

            qlabel = self.c[candidatesindexs[i]][0]
            aimg = self.c[candidatesindexs[i]][1]
            pimg = None
            nimg = None

            rank = ranks[:, i]
            for indj in range(len(self.p)):
                if indj > self.qscale:
                    continue
                j = rank[indj]
                rlabel = self.p[j][0]
                if (rlabel != qlabel) and (nimg is None):
                    nimg = self.p[j][1]
                    nlabel = rlabel
                    if indj > 2:
                        pimg = self.p[rank[indj - 1]][1]
                        plabel = self.p[rank[indj - 1]][0]
                elif (rlabel == qlabel) and (nimg is not None):
                    pimg = self.p[j][1]
                    plabel = self.p[j][0]

                if pimg is not None and nimg is not None:
                    tripletlist.append([(aimg, qlabel), (pimg, plabel), (nimg, nlabel)])
                    break

        random.shuffle(tripletlist)
        self.triplet_pool = tripletlist

    def __getitem__(self, index):
        (imgpath1, label1), (imgpath2, label2), (imgpath3, label3) = self.triplet_pool[index]
        # label1, label2, label3 = self.labels[index]
        try:
            img1 = Image.open(imgpath1).convert('RGB')
            img2 = Image.open(imgpath2).convert('RGB')
            img3 = Image.open(imgpath3).convert('RGB')
        except:
            return self.__getitem__(index + 1)

        policy = ImageNetPolicy()

        img1 = policy(img1)
        img2 = policy(img2)
        img3 = policy(img3)

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)

        return (img1, label1), (img2, label2), (img3, label3)

    def __len__(self):
        return len(self.triplet_pool)