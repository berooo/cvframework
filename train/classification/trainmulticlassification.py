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
from graph import builGraph,buildLoss
from torch.autograd import Variable
from testt.testonlinepair import parser as partest
from testt.testonlinepair import testOnlinepair
from datasets.GOGDataset import GoclassDataset
from util.Balancedparallel import DataParallelCriterion
from queue import Queue

event= threading.Event()
over=threading.Event()

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=128,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/mnt/sdb/shibaorong/data/googleLandmark/train.csv',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/googlelandmark/classification/resnet50/parameter_04.pkl',
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
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
min_loss = float("inf")
train_done = False
data_queue=Queue(10000)
datalen=0

def read_data(args):

    global datalen
    mytraindata = GoclassDataset(path=args.data_dir, autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size, shuffle=True, num_workers=100)
    datalen=len(mytrainloader)

    while not over.is_set():
        # 生成 batch 数据
        for i, (batch_x,batch_y) in enumerate(mytrainloader):
            data_queue.put([batch_x, batch_y])  # put到队列数据，超过所设个数，会阻塞此线程

        event.clear()
        while not event.is_set():  # 一个周期结束，等待事件 event.set()
            pass

def trainclassification(mymodel,epoch,optimizer,thisloss,scheduler):
    global min_loss,datalen

    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    acc_avg=AverageMeter()
    print('epoch {}'.format(epoch + 1))
    print(min_loss)
    train_loss=0.

    bar = Bar('[{}]{}'.format('classification-GGLM', 'train'), max=datalen)
    since = time.time()

    ld=data_queue.qsize()
    print(ld)

    for index in range(datalen):
      if not event.is_set():
          event.set()
      batch_x,batch_y=data_queue.get()
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
      loss=loss.mean()

      loss.backward()
      optimizer.step()



      try:
          out = torch.cat(out, dim=0)
      except Exception as e:
          print(e)

      prediction = torch.argmax(out, 1)
      train_acc += (prediction == batch_y).sum().float()
      acc = train_acc / len(batch_x)

      batch_timer.update(time.time() - since)
      since = time.time()
      prec_losses.update(loss, 1)
      acc_avg.update(acc,1)

      log_msg = ('\n[epoch:{epoch}][iter:({batch}/{size})]' +
                 '[lr:{lr}] loss: {loss:.4f}acc: {acc:.4f}   | eta: ' +
                 '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
          .format(
          epoch=epoch + 1,
          batch=index + 1,
          size=datalen,
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


def main():
    global args
    global min_loss

    args=parser.parse_args()
    cuda_gpu = torch.cuda.is_available()


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

    thread_data = threading.Thread(target=read_data, args=(args,))
    thread_data.start()
    time.sleep(0.5)
    event.set()
    while data_queue.qsize() < 5:
        pass

    for epoch in range(startepoch, args.maxepoch):
        t = threading.Thread(target=trainclassification,args=(mymodel,epoch,optimizer,thisloss,scheduler,))
        t.start()
        t.join()
        scheduler.step()

    over.set()
    event.set()
    thread_data.join()

if __name__=='__main__':
    main()
    '''if epoch==startepoch or (epoch+1)%5==0:
                print('epoch eval {}'.format(epoch))
                testargs = partest.parse_args()
                pklword = args.train_dir.split('/')[-1]
                newpkl = 'parameter_%02d.pkl' % (epoch)
                path = args.train_dir.replace(pklword, newpkl)
                print(path)
                testargs.train_dir = path
                testargs.backbone=args.backbone
                testOnlinepair(testargs, cuda_gpu)'''