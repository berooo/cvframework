import os
import sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import argparse
import json
import datetime
import torch
from datasets.commonDataset import myDataset,FolderDataset
from datasets.imageListDateset import ImagesFromList
from graph import builGraph
from option.evaluate import compute_map_and_print, compute_ap
from util.util import loadquery
import numpy as np
from network.outputdim import OUTPUT_DIM
import random
import json

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=1,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/mnt/sdb/shibaorong/data/DIGIX/test_data_B/gallery',
                    help='destination where trained network should be saved')
parser.add_argument('--valdata_dir',default='/mnt/sdb/shibaorong/data/DIGIX/test_data_B/query',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/DIGIX/resnet50_glem_arcface_embedding/model_best.pth.tar',
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=3097,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0],
                    help='destination where trained network should be saved')


result={}

def parseeye(trainpath,testpath):
  queryList=[]
  test=json.load(open(testpath))
  traindata=json.load(open(trainpath))
  sample_pool={}
  i=0
  for d in traindata:
    filename = d['filenames']
    label = d['label_id']
    if label not in sample_pool:
      sample_pool[label] = [filename]
      i += 1
    else:
      sample_pool[label].append(filename)

  for index,t in enumerate(test):
    lb=t['label_id']
    f=t['filenames']
    q={'queryimgid': f, 'ok': sample_pool[lb], 'junk': [], 'good': [], 'boxes': []}
    queryList.append(q)
  return queryList


def testmodel(args,cuda_gpu,type='extractor',similartype='dot'):
    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 type, cuda_gpu=cuda_gpu, pretrained=False)
    if os.path.exists(args.train_dir):
        print(args.train_dir)
        checkpoint = torch.load(args.train_dir,map_location='cpu')
        mymodel.load_state_dict(checkpoint['model_state_dict'])
    print(similartype)

    mydatabasedata = FolderDataset(args.data_dir,mode='test')
    mydatabaseloader = torch.utils.data.DataLoader(mydatabasedata, batch_size=args.batch_size,num_workers=20, shuffle=False)

    myquerydata = FolderDataset(args.valdata_dir, mode='test')
    myqueryloader = torch.utils.data.DataLoader(myquerydata, batch_size=args.batch_size, num_workers=20, shuffle=False)

    queryimgs=[]
    mymodel.eval()
    with torch.no_grad():
        print('>> Extracting descriptors for query images...')

        lenq = len(myqueryloader)
        qoolvecs = torch.zeros(OUTPUT_DIM[args.backbone], lenq).cuda()


        for i, (input,imgpath) in enumerate(myqueryloader):
            out= mymodel(input.cuda())
            if isinstance(out,list):
                out=torch.cat(out,dim=0)

            qoolvecs[:, i] = out.data.squeeze()
            imgpath=str(imgpath[0]).split('/')[-1]
            queryimgs.append(imgpath)

            if (i + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(i + 1, lenq), end='')
        print('')


        poolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(mydatabaseloader)).cuda()

        idlist = []

        print('>> Extracting descriptors for database images...')
        for index, data in enumerate(mydatabaseloader):
            batch_x,  batch_id = data
            idd=batch_id[0].split('/')[-1]
            idlist.append(idd)
            if cuda_gpu:
                batch_x = batch_x.cuda()

            batch_x = batch_x.float()
            out = mymodel(batch_x)

            if isinstance(out,list):
                out=torch.cat(out,dim=0)

            poolvecs[:, index] = out
            if (index + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(index + 1, len(mydatabaseloader)), end='')

        vecs = poolvecs.cpu().numpy()
        qvecs = qoolvecs.cpu().numpy()

        if similartype == 'dot':
            scores = np.dot(vecs.T, qvecs)
            ranks = np.argsort(-scores, axis=0)
        elif similartype == 'euclidean':
            dis = np.zeros([vecs.shape[1], qvecs.shape[1]])

            for j in range(qvecs.shape[1]):
                d = (vecs - np.reshape(qvecs[:, j], (qvecs.shape[0], 1))) ** 2
                disj = np.sum(d, axis=0)
                dis[:, j] = disj
            ranks = np.argsort(dis, axis=0)
        #compute_map_and_print(dataset, ranks, gnd, idlist)
        output='thissubmission.csv'
        fout = open(output, 'w')
        formats = '{0[0]},{{%s}}' % (','.join(['{0[%s]}' % str(i + 1) for i in range(10)]))
        for j in range(ranks.shape[1]):
            record=ranks[:,j]
            qrcnt=queryimgs[j].strip().split('/')[-1]
            rf_res=[idlist[i] for i in record]
            olist = [qrcnt] + [it.strip().split('/')[-1] for it in rf_res[:10]]
            out = formats.format(olist) + '\n'
            print(out)
            fout.write(out)
            # out.encode('utf-8')
        fout.close()



if __name__=='__main__':

    global args
    args = parser.parse_args()
    cuda_gpu = torch.cuda.is_available()
    testmodel(args, cuda_gpu, type='retrieval', similartype='dot')

