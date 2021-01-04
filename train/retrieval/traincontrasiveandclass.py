# -*- coding: utf-8 -*-

"""
@author: shibaorong

@contact: diamond_br@163.com

@Created on: 2020/6/2 10:06
"""
import time

from datasets.siameseData import SiameseData
import argparse
import time
from parse.genpair import genPairs
from input import *
from graph import builGraph,buildLoss
from torch.autograd import Variable
from torch.optim import lr_scheduler
from testt.testonlinepair import testOnlinepair
from testt.testonlinepair import parser as partest
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=64,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/home/shibaorong/modelTorch/out/paristrainpair.json',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/paris/contrasive/plusclass11/parameter_06.pkl',
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

min_loss = float("inf")

def trainSiamese(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,scheduer):
    global min_loss

    print('epoch {}'.format(epoch + 1))
    mymodel.train()
    trainloss=0.
    record=0
    for index, (img1,label1,img2,label2) in enumerate(mytrainloader):
      iter_start_time = time.time()
      if cuda_gpu:
        img1 = img1.cuda()
        label1=label1.cuda()
        img2 = img2.cuda()
        label2=label2.cuda()

      img1 = img1.float()
      img2 = img2.float()

      img1, img2 = Variable(img1), Variable(img2)
      n,c,h,w=img1.size()
      imgpair=torch.cat((img1,img2),axis=0)
      optimizer.zero_grad()
      out,features=mymodel(imgpair)
      out1,out2=out.split(n,dim=0)
      f=features['lastlinear']
      f1,f2=f.split(n,dim=0)
      #out1,out2=mymodel(img1,img2)
      tloss=buildLoss.ContrastiveLoss()
      target=np.zeros([len(label1),1])
      for i in range(len(label1)):
          if label1[i]==label2[i]:
              target[i,:]=1
          else:
              target[i,:]=0
      target=torch.from_numpy(target).cuda().float()
      contrasiveloss=tloss(f1,f2,target)
      c1loss=nn.CrossEntropyLoss()
      loss1=c1loss(out1,label1)
      loss2=c1loss(out2,label2)

      loss=contrasiveloss+loss1+loss2

      if loss.item()>0:
        trainloss += loss.item()
        record+=1
      loss.backward()
      optimizer.step()
      t=time.time()-iter_start_time
      if (index+1)%10==0:
        if record!=0:
          print('Train Loss: {:.6f}, Time:{:.3f}'.format(trainloss/record,t))

      torch.cuda.empty_cache()

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
                                 'classification', cuda_gpu=cuda_gpu)

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


    for epoch in range(startepoch, args.maxepoch):

        print('epoch eval {}'.format(epoch))
        testargs=partest.parse_args()
        pklword = args.train_dir.split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (epoch)
        path = args.train_dir.replace(pklword, newpkl)
        testargs.train_dir=path

        testOnlinepair(testargs, cuda_gpu)

        scheduler.step()
        genPairs(args.jsonfile,args.tofile)
        mytraindata =SiameseData(path=args.data_dir, autoaugment=args.autoaugment)
        mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size, shuffle=True)
        trainSiamese(mymodel,epoch,cuda_gpu,optimizer,mytrainloader,scheduler)


if __name__=='__main__':
    main()