# -*- coding: utf-8 -*-

"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/6/7 10:06
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
print(BASE,flush=True)

from train.adjustLR import _learning_rate_schedule
from datasets.CartoonDataset import CartoonDataset
import argparse
import time
from parse.genpair import genPairs
from input import *
from graph import builGraph,buildLoss
from torch.autograd import Variable
from torch.optim import lr_scheduler
from testt.testonlinepair import parser as partest
from testt.testonlinepair import testOnlinepair
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=256,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='../../datasets/data/train',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='../../out/contrasive/model_best.pyth',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=True,
                    help='destination where trained network should be saved')
parser.add_argument('--finetune_dir',default='/mnt/sdb/shibaorong/logs/paris/classification/trial/parameter_08.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='vgg16',
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
parser.add_argument('--jsonfile',default='/home/shibaorong/modelTorch/out/paris6ktrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--tofile',default='/home/shibaorong/modelTorch/out/paris',
                    help='destination where trained network should be saved')
min_loss = float("inf")

def trainSiamese(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,scheduer):
    global min_loss
    print('epoch {}'.format(epoch + 1))

    trainloss=0.
    record=0
    for index, (img1,img2,label1,label2,target) in enumerate(mytrainloader):

      iter_start_time = time.time()
      if cuda_gpu:
        img1 = img1.cuda()
        label1=label1.cuda()
        img2 = img2.cuda()
        label2=label2.cuda()

      img1 = img1.float()
      img2 = img2.float()

      img1, img2 = Variable(img1), Variable(img2)

      optimizer.zero_grad()
      #out1,out2=mymodel(img1,img2)
      n, c, h, w = img1.size()
      imgpair = torch.cat((img1, img2), axis=0)
      optimizer.zero_grad()
      out= mymodel(imgpair)
      out1, out2 = out.split(n, dim=0)
      tloss=buildLoss.ContrastiveLoss()
      loss=tloss(out1,out2,target)

      if loss.item()>0:
        trainloss += loss.item()
        record+=1
      loss.backward()
      optimizer.step()
      t=time.time()-iter_start_time
      if (index+1)%10==0:
        if record!=0:
          print('Train Loss: {:.6f}, Time:{:.3f}'.format(trainloss/record,t))

    if record>0:
      trainloss = trainloss / record

    print('epoch Train Loss: {:.6f}'.format(trainloss))
    pklword = args.train_dir.split('/')[-1]
    newpkl = 'parameter_%02d.pkl' % (epoch + 1)
    path = args.train_dir.replace(pklword, newpkl)

    is_best = trainloss < min_loss
    if is_best:
        min_loss=trainloss
    save_checkpoint({'epoch': epoch,
                     'model_state_dict': mymodel.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': trainloss,
                     'scheduer':scheduer
                     }, is_best, path)


def main():

    global args
    global min_loss
    args=parser.parse_args()
    cuda_gpu = torch.cuda.is_available()

    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'retrieval', cuda_gpu=cuda_gpu)
    if args.optimizer == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=args.LR)
    else:
        optimizer = torch.optim.Adam(mymodel.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    startepoch = 0
    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']+1
        min_loss = checkpoint['loss']
        if 'scheduler' in checkpoint:
            scheduler = checkpoint['scheduler']

    mytraindata = CartoonDataset(args.data_dir)
    for epoch in range(startepoch, args.maxepoch):
        _learning_rate_schedule(optimizer, epoch, args.maxepoch, args.LR)
        mytraindata.create_epoch()
        mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size, shuffle=True)
        trainSiamese(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,scheduler)


if __name__=='__main__':
    main()