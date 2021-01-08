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
    mymodel,epoch,optimizer,mytrainloader,loss_func1,loss_func2=params
    global min_loss

    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    acc_avg=AverageMeter()
    print('epoch {}'.format(epoch + 1),flush=True)
    print(min_loss,flush=True)

    bar = Bar('[{}]{}'.format('CtoP---cartoon', 'train'), max=len(mytrainloader))
    since = time.time()
    for index, (img,label,cps,_) in enumerate(mytrainloader):
      train_acc = 0.
      data_timer.update(time.time() - since)
      if torch.cuda.is_available():
          img= img.cuda()
          label = label.cuda()
          cps=cps.cuda()

      img = Variable(img.float())

      optimizer.zero_grad()

      feat,logits= mymodel(img)

      loss1=loss_func1(feat,(label,cps))

      loss2=loss_func2(logits,label)
      
      print(loss2.item(),flush=True)
      
      loss=loss2
      loss.backward()
      optimizer.step()

      prediction = torch.argmax(logits, 1)
      train_acc += (prediction == label).sum().float()
      acc = train_acc / len(label)



      batch_timer.update(time.time() - since)
      since = time.time()
      prec_losses.update(loss, 1)
      acc_avg.update(acc,1)
      log_msg = ('\n[epoch:{epoch}][iter:({batch}/{size})]' +
                 '[lr:{lr}] loss: {loss:.4f} acc: {acc:.4f} | eta: ' +
                 '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
          .format(
          epoch=epoch + 1,
          batch=index + 1,
          size=len(mytrainloader),
          lr=optimizer.param_groups[0]['lr'],
          loss=prec_losses.avg,
          acc=acc_avg.avg,
          dt=data_timer.val,
          bt=batch_timer.val,
          tt=bar.elapsed_td)
      
      print(log_msg,flush=True)
      
      index+=1
      bar.next()
      step+=1
    bar.finish()

    pklword = cfg.INPUT.CKPTPATH.split('/')[-1]
    newpkl = 'parameter_%02d.pkl' % (epoch + 1)
    path = cfg.INPUT.CKPTPATH.replace(pklword, newpkl)

    is_best = prec_losses.avg < min_loss
    if is_best:
        min_loss = prec_losses.avg

    save_checkpoint({'epoch': epoch,
                     'model_state_dict': mymodel.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': prec_losses.avg,
                     'step': step
                     }, is_best, path)

def main(cfg):

    global min_loss
    global step

    mymodel = setup_model(cfg)
    optimizer = optim.construct_optimizer(mymodel)
    dataset=process_traindir(cfg.INPUT.DATAPATH)
    mytraindata=ImageDataset(dataset,cfg)
    mytrainloader=DataLoader(mytraindata,batch_size=cfg.TRAIN.BATCH_SIZE,
                             sampler=RandomIdentitySampler(dataset,cfg.TRAIN.BATCH_SIZE,cfg.DATALOADER.NUM_INSTANCE),
                             num_workers=cfg.DATALOADER.NUM_INSTANCE, collate_fn=train_collate_fn)

    loss1= TripletLoss()
    loss2=nn.CrossEntropyLoss()

    startepoch,minloss,sstep=load_checkpoint(cfg.INPUT.CKPTPATH,mymodel,optimizer)

    min_loss=minloss
    step=sstep

    for epoch in range(startepoch, cfg.OPTIM.MAX_EPOCH):
        _learning_rate_schedule(optimizer, epoch, cfg.OPTIM.MAX_EPOCH, cfg.OPTIM.BASE_LR)
        traintricls(mymodel,epoch,optimizer,mytrainloader,loss1,loss2)


if __name__=='__main__':
    config.load_cfg_fom_args("Train a tricls model.")
    cfg.freeze()
    print(cfg,flush=True)
    main(cfg)

