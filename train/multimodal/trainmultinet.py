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
from tensorboardX import SummaryWriter
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
print(BASE)
import argparse
from torch.optim import lr_scheduler
from  progress.bar import Bar
from input import *
import time
from config import cfg,config
from losses.arcface_loss import ArcMarginLoss
from prefetch_generator import BackgroundGenerator
from graph import builGraph,buildLoss
from torch.autograd import Variable
from network.multimodal.multinet import multi_net
from graph.buildLoss import ContrastiveLoss
from datasets.CartoonDataset import CartoonDataset
from train.adjustLR import _learning_rate_schedule
from util.util import *

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=64,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/home/shibaorong/cartoon/datasets/data/cartoon',
                    help='destination where trained network should be saved')
parser.add_argument('--log_dir',default='/mnt/sdb/shibaorong/logs/cartoon/',
                    help='destination where trained network should be saved')
parser.add_argument('--finetune_dir',default='/mnt/sdb/shibaorong/logs/cartoon/checkpoints/model_best.pyth',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=True,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=123,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.001,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
min_loss = float("inf")
writer=SummaryWriter('DIGIX')
step=0


def test(*params):
    mymodel,mytrainloader = params
    mymodel.eval()
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    acc_avg = AverageMeter()
    print(min_loss,flush=True)

    bar = Bar('[{}]{}'.format('classification-Holidays', 'train'), max=len(mytrainloader))
    since = time.time()
    for index, (batch_x, batch_y) in enumerate(mytrainloader):
        train_acc = 0.
        data_timer.update(time.time() - since)
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        batch_x = batch_x.float()
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        out = mymodel(batch_x)
        # loss,prediction=mymodel(batch_x,batch_y,thisloss)

        out=torch.cat(out,dim=0)
        prediction = torch.argmax(out, 1)
        train_acc += (prediction == batch_y).sum().float()
        acc = train_acc / len(batch_x)

        batch_timer.update(time.time() - since)
        since = time.time()
        acc_avg.update(acc, 1)
        # train_loss += loss.item()

        log_msg = ('\n[iter:({batch}/{size})]' +
                   ' loss: {loss:.4f}acc: {acc:.4f}   | eta: ' +
                   '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
            .format(

            batch=index + 1,
            size=len(mytrainloader),

            loss=prec_losses.avg,
            acc=acc_avg.avg,
            dt=data_timer.val,
            bt=batch_timer.val,
            tt=bar.elapsed_td)
        print(log_msg,flush=True)
        index += 1
        bar.next()
    bar.finish()

def trainmultimodal(*params):
    global step
    mymodel, epoch, optimizer, mytrainloader,loss_contrasive,loss_crossentro,Arcloss=params
    global min_loss
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    acc_avg=AverageMeter()
    print('epoch {}'.format(epoch + 1),flush=True)
    print(min_loss,flush=True)
    train_loss=0.

    bar = Bar('[{}]{}'.format('CtoP---cartoon', 'train'), max=len(mytrainloader))
    since = time.time()
    for index, (img1,img2,label1,label2,target) in enumerate(mytrainloader):
      train_acc = 0.
      data_timer.update(time.time() - since)
      if torch.cuda.is_available():
          img1 = img1.cuda()
          label1 = label1.cuda()
          img2 = img2.cuda()
          label2 = label2.cuda()
          target=target.cuda()

      img1 = img1.float()
      img2 = img2.float()

      img1, img2= Variable(img1), Variable(img2)

      optimizer.zero_grad()

      fc,fp= mymodel(img1,img2)


      loss_contra_2 = loss_contrasive(fc, fp, target)
      out1 = Arcloss(fc, label1)
      loss1 = loss_crossentro(out1, label1)

      out2=Arcloss(fp,label2)
      loss2 = loss_crossentro(out2, label2)

      #loss,prediction=mymodel(batch_x,batch_y,thisloss)
      loss=loss1+loss2+loss_contra_2
      loss=loss.mean()

      writer.add_scalar('Train/Loss', loss, step)
      loss.backward()
      optimizer.step()

      prediction1 = torch.argmax(out1, 1)
      train_acc += (prediction1 == label1).sum().float()
      prediction2 = torch.argmax(out2, 1)
      train_acc += (prediction2 == label2).sum().float()
      acc = train_acc / (2*len(img1))

      writer.add_scalar('Train/Acc', acc, step)
      batch_timer.update(time.time() - since)
      since = time.time()
      prec_losses.update(loss, 1)
      acc_avg.update(acc,1)
      writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)
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
      writer.flush()
      bar.next()
      step+=1
    bar.finish()

    newpkl = 'parameter_%02d.pkl' % (epoch + 1)
    path = os.path.join(args.log_dir,'checkpoints',newpkl)

    is_best = prec_losses.avg < min_loss
    if is_best:
        min_loss=prec_losses.avg

    save_checkpoint({'epoch': epoch,
                       'model_state_dict': mymodel.state_dict(),
                       'arcface_state_dict':Arcloss.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': prec_losses.avg,
                       'step':step
                       }, is_best, path)

def main():
    config.load_cfg_fom_args("Train a tricls model.")
    cfg.freeze()
    global args
    global min_loss
    global step
    args=parser.parse_args()
    mytraindata = CartoonDataset(args.data_dir,cfg)

    mymodel = multi_net(modelName=args.backbone)
    print(mymodel,flush=True)
    mymodel = torch.nn.DataParallel(mymodel, device_ids=[0]).cuda()

    '''if os.path.exists(args.log_dir):
        pc=torch.load(os.path.join(args.log_dir,'c','resnet50/model_best.pyth'),map_location='cpu')
        pc_state={ k.replace('module._baselayer.0.','') : v for index,(k, v) in enumerate(pc['model_state_dict'].items()) if index<26}
        mymodel.module.branch_c.load_state_dict(pc_state)
        pp=torch.load(os.path.join(args.log_dir,'p','resnet50/model_best.pyth'),map_location='cpu')
        pp_state = {k.replace('module._baselayer.0.', ''): v for index,(k, v) in enumerate(pp['model_state_dict'].items()) if index<26}
        mymodel.module.branch_p.load_state_dict(pp_state)'''

    if args.optimizer == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=args.LR)
    else:
        optimizer = torch.optim.Adam(mymodel.parameters(),lr=args.LR)

    loss_contrasive = ContrastiveLoss()
    loss_crossentro= nn.CrossEntropyLoss()
    Arcloss = torch.nn.DataParallel(ArcMarginLoss(args.classnum), device_ids=args.gpu).cuda()
    startepoch = 0

    if os.path.exists(args.finetune_dir):
        checkpoint=torch.load(args.finetune_dir,map_location='cpu')
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        Arcloss.load_state_dict(checkpoint['arcface_state_dict'])
        startepoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['loss']
        if 'step' in checkpoint:
            step = checkpoint['step']

    for epoch in range(startepoch, args.maxepoch):
        _learning_rate_schedule(optimizer, epoch, args.maxepoch, args.LR)
        mytraindata.create_epoch()
        mytrainloader = DataLoaderX(mytraindata, batch_size=args.batch_size, shuffle=True, num_workers=0)
        trainmultimodal(mymodel,epoch,optimizer,mytrainloader,loss_contrasive,loss_crossentro,Arcloss)

    writer.close()

if __name__=='__main__':
    main()

