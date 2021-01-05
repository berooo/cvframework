# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 19:16
"""

import os
import sys
from tensorboardX import SummaryWriter
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
print(BASE,flush=True)
import argparse
from  progress.bar import Bar
import time
from network.outputdim import OUTPUT_DIM
from losses.arcface_loss import ArcMarginLoss
from graph import builGraph
from torch.autograd import Variable
from datasets.CartoonDataset import generalclsDataset
from train.adjustLR import _learning_rate_schedule
from util.util import *

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=256,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='../../datasets/data/train',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='../../out/normalcls/model_best.pyth',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=True,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='vgg16',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=124,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.001,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')

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

def trainclassification(*params):

    global step
    mymodel, epoch, optimizer, thisloss, mytrainloader, Arcloss=params
    global min_loss
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    acc_avg=AverageMeter()
    print('epoch {}'.format(epoch + 1),flush=True)
    print(min_loss,flush=True)
    train_loss=0.

    bar = Bar('[{}]{}'.format('classification-DIGIX', 'train'), max=len(mytrainloader))
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
      out=Arcloss(out,batch_y)
      loss = thisloss(out, batch_y)
      #loss,prediction=mymodel(batch_x,batch_y,thisloss)
      loss=loss.mean()
      writer.add_scalar('Train/Loss', loss, step)
      loss.backward()
      optimizer.step()

      #out=torch.cat(out,dim=0)
      prediction = torch.argmax(out, 1)
      train_acc += (prediction == batch_y).sum().float()
      acc = train_acc / len(batch_x)

      writer.add_scalar('Train/Acc', acc, step)

      batch_timer.update(time.time() - since)
      since = time.time()
      prec_losses.update(loss, 1)
      acc_avg.update(acc,1)
      #train_loss += loss.item()
      writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)
      log_msg = ('\n[epoch:{epoch}][iter:({batch}/{size})]' +
                 '[lr:{lr}] loss: {loss:.4f}acc: {acc:.4f}   | eta: ' +
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

    pklword =args.train_dir.split('/')[-1]
    newpkl = 'parameter_%02d.pkl' % (epoch + 1)
    path = args.train_dir.replace(pklword, newpkl)

    is_best = prec_losses.avg < min_loss
    if is_best:
        min_loss=prec_losses.avg

    save_checkpoint({'epoch': epoch,
                       'model_state_dict': mymodel.state_dict(),
                       'arcface_state_dict':Arcloss.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': prec_losses.avg,
                        'pmark':list(mymodel.named_parameters())[0][1][0][0][0][0],
                       'step':step
                       }, is_best, path)

def main():
    global args
    global min_loss
    global step
    args=parser.parse_args()
    cuda_gpu = torch.cuda.is_available()

    mytraindata = generalclsDataset(args.data_dir)
    mytrainloader = DataLoaderX(mytraindata, batch_size=args.batch_size, shuffle=True, num_workers=0)
    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'retrieval', cuda_gpu=cuda_gpu,pretrained=True)
    #mymodel=models.resnet50(pretrained=True).cuda()
    if args.optimizer == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=args.LR)
    else:
        optimizer = torch.optim.Adam(mymodel.parameters(),lr=args.LR)

    Arcloss=torch.nn.DataParallel(ArcMarginLoss(args.classnum,in_features=OUTPUT_DIM[args.backbone]),device_ids=args.gpu).cuda()
    thisloss = nn.CrossEntropyLoss()
    startepoch = 0


    if os.path.exists(args.train_dir):
        print(args.train_dir,flush=True)
        checkpoint = torch.load(args.train_dir,map_location='cpu')
        print(mymodel.named_parameters(),flush=True)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        Arcloss.load_state_dict(checkpoint['arcface_state_dict'])
        print(mymodel.named_parameters(),flush=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']+1
        min_loss = checkpoint['loss']
        if 'step' in checkpoint:
         step=checkpoint['step']


    for epoch in range(startepoch, args.maxepoch):
        _learning_rate_schedule(optimizer, epoch, args.maxepoch, args.LR)

        trainclassification(mymodel,epoch,optimizer,thisloss,mytrainloader,Arcloss)
        #_learning_rate_schedule(optimizer,epoch,args.maxepoch,args.LR)
    #test(mymodel,mytrainloader)
    writer.close()


if __name__=='__main__':
    main()

