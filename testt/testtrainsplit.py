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
parser.add_argument('--data_dir',default='/home/shibaorong/modelTorch/out/holidaystrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--valdata_dir',default='/home/shibaorong/modelTorch/out/holidaystest.json',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/holidays/base/resnet50_arc/parameter_150.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=False,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=500,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')
parser.add_argument('--json_file',default='/home/shibaorong/modelTorch/out/holidaystrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--height',default=224,
                    help='destination where trained network should be saved')
parser.add_argument('--width',default=224,
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
    mytraindata = myDataset(path=args.json_file, height=args.height, width=args.width,
                            autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size,num_workers=10, shuffle=False)

    gnd =parseeye(args.json_file,args.valdata_dir)
    #gnd=random.sample(gnd,50)
    mymodel.eval()
    with torch.no_grad():
        print('>> Extracting descriptors for query images...')
        qloader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[i['queryimgid'] for i in gnd], imsize=mytraindata.height,
                           transform=mytraindata.transform),
            batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        if type=='base':
            qoolvecs = torch.zeros(args.classnum, len(gnd)).cuda()
        elif type=='extractor':
            qoolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(gnd)).cuda()
        lenq = len(qloader)
        train_acc=0.
        for i, input in enumerate(qloader):
            out= mymodel(input.cuda())
            if isinstance(out,list):
                out=torch.cat(out,dim=0)

            qoolvecs[:, i] = out.data.squeeze()
            prediction = torch.argmax(out, 1)

            if (i + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(i + 1, lenq), end='')
        print('')

        if type=='extractor':
            poolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(mytrainloader)).cuda()
        elif type=='base':
            poolvecs = torch.zeros(args.classnum, len(mytrainloader)).cuda()
        idlist = []
        train_acc=0
        print('>> Extracting descriptors for database images...')
        for index, data in enumerate(mytrainloader):
            batch_x, batch_y, batch_id = data
            idlist.append(batch_id[0])
            if cuda_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            batch_x = batch_x.float()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = mymodel(batch_x)
            if isinstance(out,list):
                out=torch.cat(out,dim=0)

            prediction = torch.argmax(out, 1)
            train_acc += (prediction == batch_y).sum().float()
            acc = train_acc / len(batch_x)

            poolvecs[:, index] = out
            if (index + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(index + 1, len(mytrainloader)), end='')

        vecs = poolvecs.cpu().numpy()
        qvecs = qoolvecs.cpu().numpy()

        # search, rank, and print
        #scores = np.dot(vecs.T, qvecs)
        #ranks = np.argsort(-scores, axis=0)
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

        '''scale = [5,10,20,30,40,50,60]
        reranks = ranks
        for s in scale:
            rerankvec = np.zeros(qvecs.shape)

            for i in range(qvecs.shape[1]):
                features = np.asarray([vecs[:, j] for j in reranks[:s, i]])
                rerankvec[:, i] = np.average(features, axis=0)
            scores = np.dot(vecs.T, rerankvec) + scores
            reranks = np.argsort(-scores, axis=0)

        ranks=reranks'''
        print('RQE.....................')
        map = 0.
        mrr=0.
        nq = len(gnd)  # number of queries
        aps = np.zeros(nq)
        nempty = 0

        for i in np.arange(nq):
            qgnd = np.array(gnd[i]['ok'])

            # no positive images, skip from the average
            if qgnd.shape[0] == 0:
                aps[i] = float('nan')
                nempty += 1
                continue

            try:
                qgndj = np.array(gnd[i]['junk'])
            except:
                qgndj = np.empty(0)
            r = [idlist[j] for j in ranks[:, i]]
            # sorted positions of positive and junk images (0 based)
            pos = np.arange(ranks.shape[0])[np.in1d(r, qgnd)]
            junk = np.arange(ranks.shape[0])[np.in1d(r, qgndj)]

            k = 0;
            ij = 0;
            if len(junk):
                # decrease positions of positives based on the number of
                # junk images appearing before them
                ip = 0
                while (ip < len(pos)):
                    while (ij < len(junk) and pos[ip] > junk[ij]):
                        k += 1
                        ij += 1
                    pos[ip] = pos[ip] - k
                    ip += 1

            # compute ap
            ap = compute_ap(pos, len(qgnd))
            mr=1/(pos[0]+1)
            map = map + ap
            mrr=mrr+mr
            aps[i] = ap

            # compute precision @ k
            pos += 1  # get it to 1-based


        map = map / (nq - nempty)
        mrr=mrr/(nq-nempty)
        print(type)
        print('>> {}: mAP {:.2f}'.format('eye', np.around(map * 100, decimals=2)))
        print('>> {}: MRR {:.2f}'.format('eye', np.around(mrr * 100, decimals=2)))
        return map,mrr

if __name__=='__main__':

    global args
    args = parser.parse_args()
    cuda_gpu = torch.cuda.is_available()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    '''for i in range(16,17):
        pklword = args.train_dir.split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (i)
        path = args.train_dir.replace(pklword, newpkl)
        print(path)
        args.train_dir= path

        if not os.path.exists(path):
            continue
        result[str(i)]={}
        result[str(i)]['path']=path'''
    result[str(1)] = {}
    result[str(1)]['path'] = args.train_dir
    map,mrr = testmodel(args, cuda_gpu, type='extractor', similartype='dot')
    result[str(1)]['extractor_map'] = map
    result[str(1)]['extractor_mrr']=mrr
    map,mrr=testmodel(args, cuda_gpu,type='base',similartype='dot')
    result[str(1)]['classification_map']=map
    result[str(1)]['classification_mrr']=mrr


    json.dump(result,open(now+'holidayscls.json','w'))
