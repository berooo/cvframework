#coding:utf-8
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import json
import requests
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from util.util import *
import numpy.matlib
import torchvision.datasets as datasets
from PIL import Image,ImageFile
from util.autoaugment import ImageNetPolicy
from  datasets.onlineTripletData import OnlineTripletData
from datasets.commonDataset import myDataset
from datasets.readSfmData import TuplesDataset




def get_loader(
  train_path,
  val_path,
  stage,
  train_batch_size,
  val_batch_size,
  sample_size,
  crop_size,
  workers):

  if stage in ['finetune']:

    #for train
    prepro=[]
    prepro.append(transforms.Resize(size=sample_size))
    prepro.append(transforms.CenterCrop(size=sample_size))
    prepro.append(transforms.RandomCrop(size=crop_size,padding=0))
    prepro.append(transforms.RandomHorizontalFlip())
    prepro.append(transforms.ToTensor())
    train_transform=transforms.Compose(prepro)
    train_path=train_path

    #for val
    prepro = []
    prepro.append(transforms.Resize(size=sample_size))
    prepro.append(transforms.CenterCrop(size=crop_size))
    prepro.append(transforms.ToTensor())
    val_transform = transforms.Compose(prepro)
    val_path = val_path

  elif stage in ['keypoint']:
    #for train
    prepro = []
    prepro.append(transforms.Resize(size=sample_size))
    prepro.append(transforms.CenterCrop(size=sample_size))
    prepro.append(transforms.RandomCrop(size=crop_size, padding=0))
    prepro.append(transforms.RandomHorizontalFlip())
    # prepro.append(transforms.RandomRotation((-15, 15)))        # experimental.
    prepro.append(transforms.ToTensor())
    train_transform = transforms.Compose(prepro)
    train_path = train_path

    # for val
    prepro = []
    prepro.append(transforms.Resize(size=sample_size))
    prepro.append(transforms.CenterCrop(size=crop_size))
    prepro.append(transforms.ToTensor())
    val_transform = transforms.Compose(prepro)
    val_path = val_path

  train_dataset=datasets.ImageFolder(root=train_path,transform=train_transform)
  val_dataset = datasets.ImageFolder(root=val_path,
                                     transform=val_transform)
  # return train/val dataloader
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=workers)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=val_batch_size,
                                           shuffle=False,
                                           num_workers=workers)

  return train_loader, val_loader













if __name__=='__main__':
  pass

