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
from input import *
from  progress.bar import Bar
from graph import builGraph,buildLoss
from torch.autograd import Variable
from torch.optim import lr_scheduler
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=128,
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

def trainSiamese(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,scheduer):

    global step
    thisloss= buildLoss.ContrastiveLoss()
    global min_loss
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()
    print('epoch {}'.format(epoch + 1), flush=True)
    print(min_loss, flush=True)
    train_loss = 0.

    bar = Bar('[{}]{}'.format('classification-DIGIX', 'train'), max=len(mytrainloader))
    since = time.time()
    for index, (img1,img2,label1,label2,target) in enumerate(mytrainloader):

        data_timer.update(time.time() - since)
        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()
            target = target.cuda()

        img1 = img1.float()
        img2 = img2.float()

        img1, img2 = Variable(img1), Variable(img2)

        optimizer.zero_grad()
        n, c, h, w = img1.size()
        imgpair = torch.cat((img1, img2), axis=0)
        out = mymodel(imgpair)
        out1, out2 = out.split(n, dim=0)
        loss = thisloss(out1, out2, target)
        loss.backward()
        optimizer.step()

        batch_timer.update(time.time() - since)
        since = time.time()
        prec_losses.update(loss, 1)
        log_msg = ('\n[epoch:{epoch}][iter:({batch}/{size})]' +
                   '[lr:{lr}] loss: {loss:.4f}  | eta: ' +
                   '(data:{dt:.3f}s),(batch:{bt:.3f}s),(total:{tt:})') \
            .format(
            epoch=epoch + 1,
            batch=index + 1,
            size=len(mytrainloader),
            lr=optimizer.param_groups[0]['lr'],
            loss=prec_losses.avg,
            dt=data_timer.val,
            bt=batch_timer.val,
            tt=bar.elapsed_td)
        print(log_msg, flush=True)
        index += 1
        bar.next()
        step += 1
    bar.finish()

    pklword = args.train_dir.split('/')[-1]
    newpkl = 'parameter_%02d.pkl' % (epoch + 1)
    path = args.train_dir.replace(pklword, newpkl)

    is_best = prec_losses.avg < min_loss
    if is_best:
        min_loss = prec_losses.avg

    save_checkpoint({'epoch': epoch,
                     'model_state_dict': mymodel.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': prec_losses.avg,
                     'pmark': list(mymodel.named_parameters())[0][1][0][0][0][0],
                     'step': step
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