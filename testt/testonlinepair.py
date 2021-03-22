# -*- coding: utf-8 -*-

"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 19:04
"""
import os
import sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import argparse
import os
import torch
from datasets.commonDataset import myDataset
from graph import builGraph
from option.evaluate import compute_map_and_print
from util.util import loadquery
import numpy as np
from network.outputdim import OUTPUT_DIM
from sklearn.decomposition import PCA
from graph.function import l2n
from util.array_tool import totensor,tonumpy
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=1,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default=['/home/shibaorong/modelTorch/out/paris6ktrain.json'],
                    help='destination where trained network should be saved')
parser.add_argument('--valdata_dir',default=['/home/shibaorong/modelTorch/out/parisquery.json','/home/shibaorong/modelTorch/out/oxfordquery.json'],
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/googlelandmark/base/resnet50/parameter_11.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=False,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=81313,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')
parser.add_argument('--json_file',default=['/home/shibaorong/modelTorch/out/paris6kall.json','/home/shibaorong/modelTorch/out/oxford5ktrain.json'],
                    help='destination where trained network should be saved')
parser.add_argument('--height',default=224,
                    help='destination where trained network should be saved')
parser.add_argument('--width',default=224,
                    help='destination where trained network should be saved')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def testOnlinepair(args,cuda_gpu,type='extractor',similartype='dot'):
    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 type, cuda_gpu=cuda_gpu, pretrained=False)
    if os.path.exists(args.train_dir):
        print(args.train_dir)
        checkpoint = torch.load(args.train_dir,map_location='cpu')
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    for index,jfile in enumerate(args.json_file):

        dataset = jfile.split('/')[-1].replace("all.json", "")
        mytraindata = myDataset(path=jfile, height=args.height, width=args.width,
                                autoaugment=args.autoaugment)
        mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)
        gnd = loadquery(args.valdata_dir[index])

        mymodel.eval()
        with torch.no_grad():

            poolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(mytrainloader)).cuda()
            idlist = []
            print('>> Extracting descriptors for {} images...'.format(dataset))
            for index, data in enumerate(mytrainloader):
                batch_x, batch_y, batch_id = data
                idlist.append(batch_id[0])
                if cuda_gpu:
                    batch_x = batch_x.cuda()

                batch_x = batch_x.float()
                # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
                out = mymodel(batch_x)
                poolvecs[:, index] = out
                if (index + 1) % 10 == 0:
                    print('\r>>>> {}/{} done...'.format(index + 1, len(mytrainloader)), end='')

            qindexs = np.arange(len(mytrainloader))[np.in1d(idlist, [i['queryimgid'] for i in gnd])]
            newgnd = [idlist[i] for i in qindexs]
            g = [[i['queryimgid'] for i in gnd].index(j) for j in newgnd]
            gnd = [gnd[i] for i in g]

            vecs = poolvecs.cpu().numpy()
            '''pca = PCA(whiten=True,n_components=1000,random_state=732)
            vecst=pca.fit_transform(np.transpose(vecs))
            vecst=l2n(totensor(vecst))
            vecs=np.transpose(tonumpy(vecst))'''

            qvecs = vecs[:, qindexs]

            # search, rank, and print
            if similartype=='dot':
                scores = np.dot(vecs.T, qvecs)
                ranks = np.argsort(-scores, axis=0)
            elif similartype=='euclidean':
                dis=np.zeros([vecs.shape[1],qvecs.shape[1]])

                for j in range(qvecs.shape[1]):
                    d = (vecs - np.reshape(qvecs[:, j], (qvecs.shape[0], 1))) ** 2
                    disj = np.sum(d, axis=0)
                    dis[:, j] = disj
                ranks=np.argsort(dis,axis=0)


            compute_map_and_print(dataset, ranks, gnd, idlist)

def testnorm(args,cuda_gpu):

    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'extractor', cuda_gpu=cuda_gpu, pretrained=False)
    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    mytraindata = myDataset(path=args.json_file, height=args.height, width=args.width,
                            autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)
    gnd = loadquery(args.valdata_dir)

    mymodel.eval()
    with torch.no_grad():

        poolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(mytrainloader)).cuda()
        idlist = []
        print('>> Extracting descriptors for database images...')
        for index, data in enumerate(mytrainloader):
            batch_x, batch_y, batch_id = data
            idlist.append(batch_id[0])
            if cuda_gpu:
                batch_x = batch_x.cuda()

            batch_x = batch_x.float()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = mymodel(batch_x)
            poolvecs[:, index] = out
            if (index + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(index + 1, len(mytrainloader)), end='')


        qindexs=np.arange(len(mytrainloader))[np.in1d(idlist,[i['queryimgid'] for i in gnd])]
        newgnd=[idlist[i] for i in qindexs]
        g=[[i['queryimgid'] for i in gnd].index(j) for j in newgnd]
        gnd=[gnd[i] for i in g]

        vecs = poolvecs.cpu().numpy()
        qvecs = vecs[:,qindexs]

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)

        dataset = args.json_file.split('/')[-1].replace("all.json", "")
        compute_map_and_print(dataset, ranks, gnd, idlist)

        '''scale=[5,10,20,30,40]
        reranks=ranks
        for s in scale:
            rerankvec=np.zeros(qvecs.shape)

            for i in range(qvecs.shape[1]):
                features=np.asarray([vecs[:,j] for j in reranks[:s,i]])
                rerankvec[:,i]=np.average(features,axis=0)
            scores=np.dot(vecs.T,rerankvec)+scores
            reranks=np.argsort(-scores,axis=0)
            compute_map_and_print(dataset, reranks, gnd, idlist)'''


def testscaleavg(args,cuda_gpu):
    kwargs={'pool':'vlad'}
    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'extractor', cuda_gpu=cuda_gpu, pretrained=False,**kwargs)
    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    mytraindata = myDataset(path=args.json_file, height=args.height, width=args.width,
                            autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)
    gnd = loadquery(args.valdata_dir)

    mymodel.eval()
    with torch.no_grad():

        #poolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(mytrainloader)).cuda()
        poolvecs = torch.zeros(2450, len(mytrainloader)).cuda()
        idlist = []
        print('>> Extracting descriptors for database images...')
        for index, data in enumerate(mytrainloader):
            batch_x, batch_y, batch_id = data
            idlist.append(batch_id[0])
            if cuda_gpu:
                batch_x = batch_x.cuda()

            batch_x = batch_x.float()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = mymodel(batch_x)
            poolvecs[:, index] = out
            if (index + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(index + 1, len(mytrainloader)), end='')

        qindexs = np.arange(len(mytrainloader))[np.in1d(idlist, [i['queryimgid'] for i in gnd])]
        newgnd = [idlist[i] for i in qindexs]
        g = [[i['queryimgid'] for i in gnd].index(j) for j in newgnd]
        gnd = [gnd[i] for i in g]

        vecs = poolvecs.cpu().numpy()
        qvecs = vecs[:, qindexs]

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)

        dataset = args.json_file.split('/')[-1].replace("all.json", "")
        compute_map_and_print(dataset, ranks, gnd, idlist)

def testmultibranch(args,cuda_gpu):
    args.train_dir='/mnt/sdb/shibaorong/logs/paris/triplet/usmine/withclass_cluster11/parameter_61.pkl'

    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 'extractor', cuda_gpu=cuda_gpu, pretrained=False)

    if os.path.exists(args.train_dir):
        checkpoint = torch.load(args.train_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    mytraindata = myDataset(path=args.json_file, height=args.height, width=args.width,
                            autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)
    gnd = loadquery(args.valdata_dir)

    mymodel.eval()
    with torch.no_grad():

        poolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(mytrainloader)).cuda()
        idlist = []
        print('>> Extracting descriptors for database images...')
        for index, data in enumerate(mytrainloader):
            batch_x, batch_y, batch_id = data
            idlist.append(batch_id[0])
            if cuda_gpu:
                batch_x = batch_x.cuda()

            batch_x = batch_x.float()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = mymodel(batch_x)
            #out=torch.cat((out1,out2),-1)
            poolvecs[:, index] = out
            if (index + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(index + 1, len(mytrainloader)), end='')


        qindexs=np.arange(len(mytrainloader))[np.in1d(idlist,[i['queryimgid'] for i in gnd])]
        newgnd=[idlist[i] for i in qindexs]
        g=[[i['queryimgid'] for i in gnd].index(j) for j in newgnd]
        gnd=[gnd[i] for i in g]

        vecs = poolvecs.cpu().numpy()
        qvecs = vecs[:,qindexs]

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)

        dataset = args.json_file.split('/')[-1].replace("all.json", "")
        compute_map_and_print(dataset, ranks, gnd, idlist)


def main():
    global args
    args = parser.parse_args()
    cuda_gpu = torch.cuda.is_available()

    testOnlinepair(args,cuda_gpu)
    #testscaleavg(args,cuda_gpu)
    #testmultibranch(args,cuda_gpu)

if __name__=='__main__':
    main()
