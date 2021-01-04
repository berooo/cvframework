# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/6/7 21:15
"""
# -*- coding: utf-8 -*-

import time

from datasets.commonDataset import IDDataset
import argparse
import time

from datasets.tripletData import miningTripletData
from parse.genpair import genPairs
from input import *
from graph import builGraph,buildLoss
from graph.clustering import kmeansCluster
from torch.autograd import Variable
from torch.optim import lr_scheduler
from network import OUTPUT_DIM
from testt.testonlinepair import testOnlinepair
from testt.testonlinepair import parser as partest

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=16,
                    help='destination where trained network should be saved')
parser.add_argument('--valdata_dir',default='/home/shibaorong/modelTorch/out/parisquery.json',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/paris/triplet/usmine/withclass_cluster11/parameter_61.pkl',
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
parser.add_argument('--jsonfile',default='/home/shibaorong/modelTorch/out/paris6ktrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--tofile',default='/home/shibaorong/modelTorch/out/paris',
                    help='destination where trained network should be saved')
parser.add_argument('--height',default=224,
                    help='destination where trained network should be saved')
parser.add_argument('--width',default=224,
                    help='destination where trained network should be saved')

min_loss = float("inf")

def trainUWCS(mymodel,epoch,cuda_gpu,optimizer,mytraindata,scheduer,clusterinfo):
    global min_loss
    print('epoch {}'.format(epoch + 1))

    trainloss=0.
    record=0
    mytraindata.create_triplet(mymodel,clusterinfo,args)

    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size, shuffle=True)

    mymodel.train()
    for index, ((img1,label1), (img2,label2), (img3,label3)) in enumerate(mytrainloader):
        iter_start_time = time.time()
        if cuda_gpu:
            img1 = img1.cuda()
            label1=label1.cuda()
            img2 = img2.cuda()
            label2=label2.cuda()
            img3 = img3.cuda()
            label3=label3.cuda()
        img1 = img1.float()
        img2 = img2.float()
        img3 = img3.float()
        img1, img2, img3 = Variable(img1), Variable(img2), Variable(img3)

        optimizer.zero_grad()
        n, c, h, w = img1.size()
        imgpair = torch.cat((img1, img2,img3), axis=0)
        try:
            out, features = mymodel(imgpair)
            out1, out2,out3 = out.split(n, dim=0)
            #f = features['lastlinear']
            f1, f2,f3 = features.split(n, dim=0)

            '''features = mymodel(imgpair)
            f1, f2, f3 = features.split(n, dim=0)'''
            tloss = buildLoss.TripletLoss()
            tripletloss = tloss(f1, f2, f3)
            c1loss = nn.CrossEntropyLoss()
            loss1 = c1loss(out1, label1)
            loss2 = c1loss(out2, label2)
            loss3 = c1loss(out3, label3)

            loss = tripletloss+(loss1+loss2+loss3)/3

            if loss.item() > 0:
                trainloss += loss.item()
                record += 1
            loss.backward()
            optimizer.step()
            t = time.time() - iter_start_time
            if (index + 1) % 10 == 0:
                if record != 0:
                    print('Train Loss: {:.6f}, Time:{:.3f}'.format(trainloss / record, t))

            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            continue

    if record > 0:
        trainloss = trainloss / record

    print('epoch Train Loss: {:.6f}'.format(trainloss))
    pklword = args.train_dir.split('/')[-1]
    newpkl = 'parameter_%02d.pkl' % (epoch + 1)
    path = args.train_dir.replace(pklword, newpkl)

    is_best = trainloss < min_loss
    if is_best:
        min_loss = trainloss
    save_checkpoint({'epoch': epoch,
                     'model_state_dict': mymodel.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': trainloss,
                     'scheduer': scheduer,
                     'clusterinfo':clusterinfo
                     }, is_best, path)


def main():
    global args
    global min_loss
    args=parser.parse_args()
    cuda_gpu = torch.cuda.is_available()

    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'multitrain', cuda_gpu=cuda_gpu)
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
        if 'clusterinfo' in checkpoint:
            clusterinfo=checkpoint['clusterinfo']


    for epoch in range(startepoch, args.maxepoch):

        #if epoch!=startepoch:
        print('epoch eval {}'.format(epoch))
        testargs = partest.parse_args()
        pklword = args.train_dir.split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (epoch)
        path = args.train_dir.replace(pklword, newpkl)
        testargs.train_dir = path

        testOnlinepair(testargs, cuda_gpu)

        scheduler.step()
        mytraindata = miningTripletData(args)
        if epoch==startepoch or epoch%5==0:

            kms=kmeansCluster(args.jsonfile,args.classnum)

            clusterinfo=kms.clustering(mymodel,outdim=OUTPUT_DIM[args.backbone])

        trainUWCS(mymodel,epoch,cuda_gpu,optimizer,mytraindata,scheduler,clusterinfo)


if __name__=='__main__':
    main()