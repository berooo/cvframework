# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 19:16
"""
import os
import copy
import threading
import sys
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import argparse
from torch.optim import lr_scheduler
from  progress.bar import Bar
from input import *
import time
from prefetch_generator import BackgroundGenerator
from graph import builGraph,buildLoss
from torch.autograd import Variable
from testt.testonlinepair import parser as partest
from testt.testonlinepair import testOnlinepair
from datasets.GOGDataset import GoclassDataset
from util.Balancedparallel import DataParallelCriterion
from util.util import *

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=64,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/mnt/sdb/shibaorong/data/googleLandmark/train.csv',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/googlelandmark/classification/resnet50/parameter_06.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=True,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=81313,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
min_loss = float("inf")


def trainclassification(mymodel,epoch,optimizer,thisloss,mytrainloader,scheduler):
    global min_loss
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    acc_avg=AverageMeter()
    print('epoch {}'.format(epoch + 1))
    print(min_loss)
    train_loss=0.

    bar = Bar('[{}]{}'.format('classification-GGLM', 'train'), max=len(mytrainloader))
    since = time.time()
    for index,(batch_x,batch_y) in enumerate(mytrainloader):
      train_acc = 0.
      data_timer.update(time.time() - since)
      if torch.cuda.is_available():
          batch_x = batch_x.cuda()
          batch_y = batch_y.cuda()
      batch_x = batch_x.float()
      batch_x, batch_y = Variable(batch_x), Variable(batch_y)
      optimizer.zero_grad()
      out = mymodel(batch_x)

      loss = thisloss(out, batch_y)
      #loss,prediction=mymodel(batch_x,batch_y,thisloss)
      loss=loss.mean()

      loss.backward()
      optimizer.step()

      #out=torch.cat(out,dim=0)
      prediction = torch.argmax(out, 1)
      train_acc += (prediction == batch_y).sum().float()
      acc = train_acc / len(batch_x)

      batch_timer.update(time.time() - since)
      since = time.time()
      prec_losses.update(loss, 1)
      acc_avg.update(acc,1)
      #train_loss += loss.item()

      log_msg = ('\n[epoch:{epoch}][iter:({batch}/{size})]' +
                 '[lr:{lr}] loss: {loss:.4f}acc: {acc:.4f}   | eta: ' +
                 '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
          .format(
          epoch=epoch + 1,
          batch=index + 1,
          size=len(mytrainloader),
          lr=scheduler.get_lr()[0],
          loss=prec_losses.avg,
          acc=acc_avg.avg,
          dt=data_timer.val,
          bt=batch_timer.val,
          tt=bar.elapsed_td)
      print(log_msg)
      index+=1
      bar.next()
    bar.finish()

    pklword =args.train_dir.split('/')[-1]
    newpkl = 'parameter_%02d.pkl' % (epoch + 1)
    path = args.train_dir.replace(pklword, newpkl)

    is_best = train_loss < min_loss
    if is_best:
        min_loss=train_loss
    save_checkpoint({'epoch': epoch,
                       'model_state_dict': mymodel.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': train_loss,
                     'scheduler': scheduler
                       }, is_best, path)

def main():
    global args
    global min_loss

    args=parser.parse_args()
    cuda_gpu = torch.cuda.is_available()

    mytraindata = GoclassDataset(path=args.data_dir, autoaugment=args.autoaugment)
    mytrainloader = DataLoaderX(mytraindata, batch_size=args.batch_size, shuffle=True, num_workers=15)
    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'classification', cuda_gpu=cuda_gpu,pretrained=True)
    if args.optimizer == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=args.LR)
    else:
        optimizer = torch.optim.Adam(mymodel.parameters())
    thisloss = DataParallelCriterion(torch.nn.CrossEntropyLoss())
    startepoch = 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir,map_location='cpu')
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']+1
        min_loss = checkpoint['loss']
        if 'scheduler' in checkpoint:
            scheduler=checkpoint['scheduler']

    for epoch in range(startepoch, args.maxepoch):
        trainclassification(mymodel,epoch,optimizer,thisloss,mytrainloader,scheduler)
        scheduler.step()


if __name__=='__main__':
    main()
