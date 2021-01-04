import time

from torch.autograd import Variable
from datasets.GOGDataset import GoDataset

import argparse
import time
from util.util import AverageMeter
from torch.optim import lr_scheduler
from parse.genpair import genTriplet
from input import *
from graph import builGraph,buildLoss
from torch.autograd import Variable
from  progress.bar import Bar
from testt.testonlinepair import parser as partest
from testt.testonlinepair import testOnlinepair
from testt.testcontrasive import testmodel

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=32,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/mnt/sdb/shibaorong/data/googleLandmark/train.csv',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/googlelandmark/triplet/parameter_04.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=False,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet101',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=2048,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')
parser.add_argument('--jsonfile',default='/home/shibaorong/modelTorch/out/pathologicaltrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--tofile',default='/home/shibaorong/modelTorch/out/pathological',
                    help='destination where trained network should be saved')

min_loss = float("inf")

def trainTriplet(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,scheduler):
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    prec_losses = AverageMeter()

    global min_loss
    print('epoch {},len {}'.format(epoch + 1,len(mytrainloader)))
    print(min_loss)
    trainloss=0.
    record=0
    bar = Bar('[{}]{}'.format('triplet-GGLM', 'train'), max=len(mytrainloader))
    since = time.time()
    for index, (img1,img2,img3) in enumerate(mytrainloader):
      data_timer.update(time.time() - since)
      iter_start_time = time.time()
      if cuda_gpu:
        img1 = img1.cuda()
        img2 = img2.cuda()
        img3=img3.cuda()

      img1 = img1.float()
      img2 = img2.float()
      img3 = img3.float()
      img1, img2,img3 = Variable(img1), Variable(img2), Variable(img3)

      optimizer.zero_grad()
      n, c, h, w = img1.size()
      imgpair = torch.cat((img1, img2, img3), axis=0)
      try:
          out= mymodel(imgpair)
          out1, out2, out3 = out.split(n, dim=0)
          tloss=buildLoss.TripletLoss()
          loss=tloss(out1,out2,out3)
      except Exception as e:
          print(e)
      batch_timer.update(time.time() - since)
      since = time.time()
      prec_losses.update(loss, 1)
      if loss.item()>0:
        trainloss += loss.item()
        record+=1
      loss.backward()
      optimizer.step()
      t=time.time()-iter_start_time
      if (index+1)%100==0:
        if record!=0:
            pass
          #print('Train Loss: {:.6f}, Time:{:.3f}'.format(trainloss/record,t))
      torch.cuda.empty_cache()
      log_msg = ('\n[epoch:{epoch}][iter:({batch}/{size})]' +
                 '[lr:{lr}] loss: {loss:.4f}  | eta: ' +
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
      bar.next()
    bar.finish()
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
                     'scheduler':scheduler
                     }, is_best, path)


def main():
    global args
    global min_loss
    args=parser.parse_args()
    cuda_gpu = torch.cuda.is_available()


    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'classification', cuda_gpu=cuda_gpu)
    if args.optimizer == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=args.LR)
    else:
        optimizer = torch.optim.Adam(mymodel.parameters())

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    startepoch = 0
    torch.cuda.empty_cache()
    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']+1
        min_loss = checkpoint['loss']
        if 'scheduler' in checkpoint:
            scheduler=checkpoint['scheduler']
    mytraindata = GoDataset(path=args.data_dir, autoaugment=args.autoaugment)
    for epoch in range(startepoch, args.maxepoch):
        if epoch==startepoch or (epoch+1)%5==0:
            testargs = partest.parse_args()
            pklword = args.train_dir.split('/')[-1]
            newpkl = 'parameter_%02d.pkl' % (epoch)
            path = args.train_dir.replace(pklword, newpkl)
            print(path)
            testargs.train_dir = path
            testargs.classnum=args.classnum
            testargs.backbone=args.backbone
            #testmodel(mymodel,testargs,cuda_gpu)
            testOnlinepair(testargs, cuda_gpu)
        mytraindata.create_triplet(mymodel,args.classnum)
        mymodel.train()
        mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size, shuffle=True)
        trainTriplet(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,scheduler)
        scheduler.step()

if __name__=='__main__':
    main()