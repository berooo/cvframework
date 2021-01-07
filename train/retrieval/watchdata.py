# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 19:16
"""

import os
import copy
import threading
from torchvision import models
import sys
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
print(BASE,flush=True)
import argparse
from torch.optim import lr_scheduler
from  progress.bar import Bar
from input import *
import time
from torch import nn
from core import optimizer as optim
from losses.arcface_loss import ArcMarginLoss
from torch.autograd import Variable
from network.retrieval.basebackbone import Base
from graph.buildLoss import ContrastiveLoss
from datasets.CartoonDataset import ImageDataset,process_traindir,train_collate_fn
from train.adjustLR import _learning_rate_schedule
from util.util import *
from config import cfg,config
from datasets.sampler import RandomIdentitySampler
from torch.utils.data import DataLoader
from losses.batchTripletLoss import TripletLoss
from core.checkpoint import load_checkpoint

min_loss = float("inf")
step=0

def setup_model(cfg):
    model=Base(cfg)
    print(model,flush=True)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    return model

def traintricls(*params):
    global step
    epoch,mytrainloader=params
    global min_loss


    print('epoch {}'.format(epoch + 1),flush=True)
    print(min_loss,flush=True)

    for index, (img,label,cps,names) in enumerate(mytrainloader):
        for i in range(img.size(0)):
            print(label[i],names[i],flush=True)
        print('----------------batchover---------------------')

def main(cfg):

    global min_loss
    global step

    dataset=process_traindir(cfg.INPUT.DATAPATH)
    mytraindata=ImageDataset(dataset,cfg)
    mytrainloader=DataLoader(mytraindata,batch_size=cfg.TRAIN.BATCH_SIZE,
                             sampler=RandomIdentitySampler(dataset,cfg.TRAIN.BATCH_SIZE,cfg.DATALOADER.NUM_INSTANCE),
                             num_workers=cfg.DATALOADER.NUM_INSTANCE, collate_fn=train_collate_fn)


    for epoch in range(0, cfg.OPTIM.MAX_EPOCH):
        traintricls(epoch,mytrainloader)


if __name__=='__main__':
    config.load_cfg_fom_args("Train a tricls model.")
    cfg.freeze()
    print(cfg,flush=True)
    main(cfg)

