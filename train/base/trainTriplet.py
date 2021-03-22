# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/6/7 21:15
"""
# -*- coding: utf-8 -*-
import os
import time
import sys
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
print(BASE)
import argparse
import time

from datasets.tripletData import miningTripletData
from parse.genpair import genPairs
from input import *
from  progress.bar import Bar
from graph import builGraph,buildLoss
from graph.clustering import kmeansCluster
from torch.autograd import Variable
from torch.optim import lr_scheduler
from network import OUTPUT_DIM
from testt.testtrainsplit import testmodel
from testt.testtrainsplit import parser as partest
from util.Balancedparallel import DataParallelCriterion
from datasets.util import eyetransfrom
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=64,
                    help='destination where trained network should be saved')
parser.add_argument('--valdata_dir',default='/home/shibaorong/modelTorch/out/eyetest.json',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/eye/tri/m1/single/parameter_61.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--finetune_dir',default='/mnt/sdb/shibaorong/logs/eye/parameter_13.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=True,
                    help='destination where trained network should be saved')
parser.add_argument('--balancedgpu',default=False,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=5,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='sgd',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')
parser.add_argument('--jsonfile',default='/home/shibaorong/modelTorch/out/eyetrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--tofile',default='/home/shibaorong/modelTorch/out/eye',
                    help='destination where trained network should be saved')
parser.add_argument('--height',default=224,
                    help='destination where trained network should be saved')
parser.add_argument('--width',default=224,
                    help='destination where trained network should be saved')

min_loss = float("inf")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def traintriandcls(mymodel,epoch,cuda_gpu,optimizer,mytraindata,scheduler):
    global min_loss
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    acc_avg = AverageMeter()
    print('epoch {}'.format(epoch + 1))

    trainloss=0.
    record=0
    mytraindata.create_triplet_classbased_1(mymodel,args)

    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size, shuffle=True,num_workers=50)

    mymodel.train()

    bar = Bar('[{}]{}'.format('base-GGLM', 'train'), max=len(mytrainloader))
    since = time.time()

    for index, ((img1,label1), (img2,label2), (img3,label3)) in enumerate(mytrainloader):
        data_timer.update(time.time() - since)
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
        try:

            o1=mymodel(img1)
            o2=mymodel(img2)
            o3=mymodel(img3)

            out1=o1['out'];f1=o1['feature'];
            out2=o2['out'];f2=o2['feature'];
            out3=o3['out'];f3=o3['feature'];

            if args.balancedgpu:
                tloss = DataParallelCriterion(buildLoss.TripletLoss())

            else:
                tloss=buildLoss.TripletLoss()
            tripletloss1 = tloss(f1, f2, f3)
            #tripletloss2=tloss(out1,out2,out3)
            #loss = 0.8*tripletloss1+0.2*tripletloss2
            loss=tripletloss1

            if loss.item() > 0:
                trainloss += loss.item()
                record += 1
            loss.backward()
            optimizer.step()
            batch_timer.update(time.time() - since)
            since = time.time()
            prec_losses.update(loss, 1)
            log_msg = ('\n[epoch:{epoch}][iter:({batch}/{size})]' +
                       '[lr:{lr}] loss: {loss:.4f}| eta: ' +
                       '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
                .format(
                epoch=epoch + 1,
                batch=index + 1,
                size=len(mytrainloader),
                lr=scheduler.get_lr()[0],
                loss=prec_losses.avg,
                dt=data_timer.val,
                bt=batch_timer.val,
                tt=bar.elapsed_td)
            print(log_msg)

        except Exception as e:
            print(e)
            continue
        index += 1
        bar.next()
    bar.finish()


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
                     'scheduer': scheduler
                     }, is_best, path)


def main():
    global args
    global min_loss
    args=parser.parse_args()
    cuda_gpu = torch.cuda.is_available()

    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'multitrain', cuda_gpu=cuda_gpu)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=args.LR)
    else:
        optimizer = torch.optim.Adam(mymodel.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    startepoch = 0
    if os.path.exists(args.finetune_dir):
        checkpoint = torch.load(args.finetune_dir,map_location='cpu')
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']+1
        min_loss = checkpoint['loss']
        if 'scheduler' in checkpoint:
            scheduler = checkpoint['scheduler']

    for epoch in range(startepoch, args.maxepoch):

        if  (epoch+1)%5==0:
            print('epoch eval {}'.format(epoch))
            testargs = partest.parse_args()
            pklword = args.train_dir.split('/')[-1]
            newpkl = 'parameter_%02d.pkl' % (epoch)
            path = args.train_dir.replace(pklword, newpkl)
            print(path)
            testargs.train_dir = path
            testargs.json_file=args.jsonfile
            testargs.valdata_dir=args.valdata_dir
            testargs.gpu=args.gpu
            testargs.classnum=args.classnum
            testmodel(testargs, cuda_gpu,type='base')


        mytraindata = miningTripletData(args,transform=eyetransfrom(args.height))

        traintriandcls(mymodel,epoch,cuda_gpu,optimizer,mytraindata,scheduler)
        scheduler.step()

if __name__=='__main__':
    main()
