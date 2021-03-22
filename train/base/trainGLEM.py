# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 19:16
"""

import argparse
import time
from google.protobuf import json_format
from torch.optim import lr_scheduler

from protos.train_pb2 import TrainConfig
from protos.model_pb2 import ModelConfig
from parse.genpair import genTriplet
from input import *
from graph import builGraph,buildLoss
from torch.autograd import Variable
from testt.testonlinepair import parser as partest
from testt.testonlinepair import testOnlinepair

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=1,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/home/shibaorong/modelTorch/out/paris6ktrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/paris/glem/parameter_17.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=True,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=11,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')


min_loss = float("inf")

def trainGLEM(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,thisloss,scheduler):
    global min_loss
    lendata=len(mytrainloader)
    print('epoch {}'.format(epoch + 1))
    print(min_loss)
    train_loss=0.
    record=0
    trainacc = 0.
    for index,(batch_x,batch_y,path) in enumerate(mytrainloader):
      record=index
      train_acc = 0.

      if cuda_gpu:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
      batch_x = batch_x.float()
      batch_x, batch_y = Variable(batch_x), Variable(batch_y)
      optimizer.zero_grad()
      out = mymodel(batch_x,need_feature=False)

      loss = thisloss(out, batch_y)
      train_loss += loss.item()

      loss.backward()
      optimizer.step()

      prediction = torch.argmax(out, 1)
      train_acc += (prediction == batch_y).sum().float()
      acc=train_acc/len(batch_x)
      trainacc+=acc
      if (index+1)%10==0:
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(loss,acc))
    train_loss = train_loss / lendata
    train_acc=trainacc/(record+1)

    print('epoch Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss, train_acc))
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

    mytraindata = myDataset(path=args.data_dir, autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size, shuffle=True)
    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'glem', cuda_gpu=cuda_gpu,pretrained=True)
    for para in mymodel.module._baselayer.parameters():
        para.requires_grad = False
    if args.optimizer == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=args.LR)
    else:
        #optimizer = torch.optim.Adam(mymodel.parameters())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mymodel.parameters()))
    thisloss = torch.nn.CrossEntropyLoss()
    startepoch = 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']+1
        min_loss = checkpoint['loss']
        if 'scheduler' in checkpoint:
            scheduler=checkpoint['scheduler']


    for epoch in range(startepoch, args.maxepoch):
        scheduler.step()
        print('epoch eval {}'.format(epoch))
        testargs = partest.parse_args()
        pklword = args.train_dir.split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (epoch)
        path = args.train_dir.replace(pklword, newpkl)
        print(path)
        testargs.train_dir = path
        if epoch!=startepoch:
            testOnlinepair(testargs, cuda_gpu,type='glemextractor')

        trainGLEM(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,thisloss,scheduler)


if __name__=='__main__':
    main()