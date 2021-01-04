# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 23:06
"""
# -*- coding: utf-8 -*-
from datasets.tripletData import TripletData

"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 19:04
"""

import argparse
import os
import torch
from datasets.commonDataset import myDataset
from datasets.imageListDateset import ImagesFromList
from graph import builGraph
from option.evaluate import compute_map_and_print
from util.util import loadquery
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=64,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/home/shibaorong/modelTorch/out/paris6ktrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--valdata_dir',default='/home/shibaorong/modelTorch/out/parisquery.json',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/home/shibaorong/modelTorch/logs/paris/offlinetriplet/plusclass/model_best.pth.tar',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=False,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=12,
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
def testtriplet(args,mymodel,cuda_gpu):

    mytraindata =  myDataset(path=args.data_dir, autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)
    gnd = loadquery(args.valdata_dir)


    if os.path.exists(args.train_dir):
        checkpoint=torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    mymodel.eval()
    with torch.no_grad():
        print('>> Extracting descriptors for query images...')
        qloader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[i['queryimgid'] for i in gnd], imsize=mytraindata.height,
                           transform=mytraindata.transform),
            batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )
        qoolvecs = torch.zeros(args.classnum, len(gnd)).cuda()
        lenq=len(qloader)
        for i, input in enumerate(qloader):
            out,_,_= mymodel(input.cuda(),input.cuda(),input.cuda())
            qoolvecs[:, i] = out.data.squeeze()
            if (i + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(i + 1, lenq), end='')
        print('')

        poolvecs = torch.zeros(args.classnum, len(mytrainloader)).cuda()
        idlist=[]
        for index, data in enumerate(mytrainloader):
            batch_x, batch_y, batch_id = data
            idlist.append(batch_id[0])
            if cuda_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            batch_x = batch_x.float()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out,_,_= mymodel(batch_x,batch_x,batch_x)
            poolvecs[:,index]=out

        vecs = poolvecs.cpu().numpy()
        qvecs = qoolvecs.cpu().numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        dataset = args.data_dir.split('/')[-1].replace("train.json", "")
        compute_map_and_print(dataset, ranks, gnd,idlist)

def main():
    global args
    args = parser.parse_args()

    cuda_gpu = torch.cuda.is_available()

    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'triplet', cuda_gpu=cuda_gpu)

    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    testtriplet(args,mymodel,cuda_gpu)

if __name__=='__main__':
    main()