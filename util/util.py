#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: util.py.py
@time: 2019/9/2 18:25
@desc:
'''

import json
import os
import shutil
import socket
import torch
import torch.nn as nn
from PIL import Image
from skimage import transform
import random as rd
from random import randint
from torch.utils.data import DataLoader
from torchvision.datasets.folder import pil_loader, accimage_loader
from prefetch_generator import BackgroundGenerator
from matplotlib import pyplot as plt

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    """
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


def htime(c):
    c = round(c)

    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)

def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
            # p.requires_grad = False
def save_checkpoint(state, is_best, path):

    torch.save(state, path)
    newpkl = path.split('/')[-1]
    best_path=path.replace(newpkl,'model_best.pyth')
    if is_best:
        shutil.copyfile(path, best_path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def loadquery(jsonpath):
  data=json.load(open(jsonpath))
  gndList=[]
  for d in data:
    for record in data[d]['good']:
      data[d]['ok'].append(record)
    data[d]['good']=[]
    gndList.append(data[d])
  return gndList

def imresize(img, imsize):
    img=img.resize((imsize, imsize))
    return img

def randomSelect(item,choicelength):
    index = [i for i in range(len(item))]
    rd.shuffle(index)
    index = index[0:choicelength]
    if type(item) is list:
        res = []
        for i in index:
            res.append(item[i])

    elif type(item) is dict:
        res = {}
        for iindex, i in enumerate(item):
            if iindex in index:
                res[i] = item[i]

    return res


def to_Onehot(label,classNum):
    onehot = torch.zeros(len(label),classNum)
    for index, by in enumerate(label):
        onehot[index, by] = 1
    return onehot

def toBinaryString(binary_like_values):
    binary_like_values=binary_like_values
    numofImage,bit_length=binary_like_values.shape
    list_string_binary=[]
    for i in range(numofImage):
        str=''
        for j in range(bit_length):
            if binary_like_values[i][j]<0:
                str+='0'
            else:
                str+='1'
        list_string_binary.append(str)
    return list_string_binary

def put_2darray(_2darray, li):
    _li = _2darray.tolist()
    for line in _li:
        li.append(line)

def save(id_list,feature_list,relu_ip1_list,label_list,file_nm):

    lis=[]
    for i in range(len(id_list)):
        s=''
        for j in range(len(relu_ip1_list[i])):
          s+=str(relu_ip1_list[i][j])
          s+=','
        li=list(zip(['id','feature','relu_ip1','label'],[id_list[i],feature_list[i],s,label_list[i]]))
        lis.append(li)
        
    with open(file_nm,'w',encoding='utf-8') as file:
        file.write(json.dumps(lis))


class BBoxCrop(object):
    def __call__(self, image,landmarks,x1,y1,x2,y2):
        h,w=image.shape[:2]

        top=y1
        left=x1
        new_h=y2-y1
        new_w=x2-x1
        image=image[top:top+new_h,
                    left:left+new_w]

        landmarks=landmarks-[left,top]

        return image,landmarks


class Rescale(object):
    def __init__(self,outputsize):
        assert isinstance(outputsize,(int,tuple))
        self.outputsize=outputsize

    def __call__(self,image,landmarks):
        h,w=image.shape[:2]
        if isinstance(self.outputsize,int):
            if h>w:
                new_h,new_w=self.outputsize*h/w,self.outputsize
            else:
                new_h,new_w=self.outputsize,self.outputsize*w/h
        else:
            new_h,new_w=self.outputsize

        new_h,new_w=int(new_h),int(new_w)
        img=transform.resize(image,(new_h,new_w),mode='constant')
        landmarks=landmarks*[new_w/w,new_h/h]

        return img,landmarks

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, input):
    return input.view(input.size(0), -1)



class SpatialAttention2d(nn.Module):

    def __init__(self,in_c,act_fn='relu'):
        super(SpatialAttention2d,self).__init__()
        self.conv1=nn.Conv2d(in_c,512,1,1)
        if act_fn.lower() in ['relu']:
            self.act1=nn.ReLU()
        elif act_fn.lower() in ['leakyrelu','leaky', 'leaky_relu']:
            self.act1=nn.LeakyReLU()
            pass
        self.conv2=nn.Conv2d(512,1,1,1)
        self.softplus=nn.Softplus(beta=1,threshold=20)

    def forward(self,x):
        x=self.conv1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.softplus(x)
        return x

    def __repr__(self):
        return self.__class__.__name__

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    return x

def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

def grid_positions(h, w, device, matrix=False):
    lines = torch.arange(
        0, h, device=device
    ).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(
        0, w, device=device
    ).view(1, -1).float().repeat(h, 1)
    if matrix:
        return torch.stack([lines, columns], dim=0)
    else:
        return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)

def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos

def savefig(filepath, fig=None, dpi=None):
    # TomNorway - https://stackoverflow.com/a/53516034
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(filepath, pad_inches=0, bbox_inches='tight', dpi=dpi)

def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos

def imshow_image(image, preprocessing=None):
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        mean = np.array([103.939, 116.779, 123.68])
        image = image + mean.reshape([3, 1, 1])
        # RGB -> BGR
        image = image[:: -1, :, :]
    elif preprocessing == 'torch':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std.reshape([3, 1, 1]) + mean.reshape([3, 1, 1])
        image *= 255.0
    else:
        raise ValueError('Unknown preprocessing parameter.')
    image = np.transpose(image, [1, 2, 0])
    image = np.round(image).astype(np.uint8)
    return image