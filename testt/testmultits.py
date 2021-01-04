import os
import sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import argparse
import json
import datetime
import torch
from datasets.commonDataset import myDataset
from datasets.imageListDateset import ImagesFromList
from graph import builGraph
from option.evaluate import compute_map_and_print, compute_ap
from util.util import loadquery,get_host_ip
import numpy as np
from network.outputdim import OUTPUT_DIM
import random
import json

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=1,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='/home/shibaorong/modelTorch/out/eyetrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--valdata_dir',default='/home/shibaorong/modelTorch/out/eyetest.json',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='/mnt/sdb/shibaorong/logs/eye/both/m1/parameter_14.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=False,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='resnet50',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=5,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')
parser.add_argument('--json_file',default='/home/shibaorong/modelTorch/out/eyetrain.json',
                    help='destination where trained network should be saved')
parser.add_argument('--height',default=224,
                    help='destination where trained network should be saved')
parser.add_argument('--width',default=224,
                    help='destination where trained network should be saved')


mmap=['classiffication','extractor','intersect']
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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


def testmodel(args,cuda_gpu,type='multitrain',similartype='dot'):
    res={}
    mymodel = builGraph.getModel(args.backbone, args.classnum, args.gpu,
                                 type, cuda_gpu=cuda_gpu, pretrained=False)
    if os.path.exists(args.train_dir):
        print(args.train_dir)
        checkpoint = torch.load(args.train_dir,map_location='cpu')
        mymodel.load_state_dict(checkpoint['model_state_dict'])
    print(similartype)
    mytraindata = myDataset(path=args.json_file, height=args.height, width=args.width,
                            autoaugment=args.autoaugment)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=args.batch_size,num_workers=50, shuffle=False)

    gnd =parseeye(args.json_file,args.valdata_dir)

    mymodel.eval()
    with torch.no_grad():
        print('>> Extracting descriptors for query images...')
        qloader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[i['queryimgid'] for i in gnd], imsize=mytraindata.height,
                           transform=mytraindata.transform),
            batch_size=args.batch_size, shuffle=False, num_workers=50, pin_memory=True
        )

        cqoolvecs = torch.zeros(args.classnum, len(gnd)).cuda()
        eqoolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(gnd)).cuda()
        iqoolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(gnd)).cuda()

        lenq = len(qloader)
        for i, input in enumerate(qloader):
            out= mymodel(input.cuda())
            if isinstance(out,list):
                out=torch.cat(out,dim=0)

            cqoolvecs[:, i] = out['out']
            eqoolvecs[:,i]=out['feature']
            iqoolvecs[:,i]=out['intersect']
            if (i + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(i + 1, lenq), end='')
        print('')


        epoolvecs = torch.zeros(OUTPUT_DIM[args.backbone], len(mytrainloader)).cuda()
        ipoolvecs=torch.zeros(OUTPUT_DIM[args.backbone], len(mytrainloader)).cuda()
        cpoolvecs = torch.zeros(args.classnum, len(mytrainloader)).cuda()

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
            if isinstance(out,list):
                out=torch.cat(out,dim=0)
            cpoolvecs[:, index] = out['out']
            epoolvecs[:,index]=out['feature']
            ipoolvecs[:, index] = out['intersect']
            if (index + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(index + 1, len(mytrainloader)), end='')

        cvecs = cpoolvecs.cpu().numpy()
        evecs=epoolvecs.cpu().numpy()
        ivecs=ipoolvecs.cpu().numpy()
        cqvecs = cqoolvecs.cpu().numpy()
        eqvecs=eqoolvecs.cpu().numpy()
        iqvecs=iqoolvecs.cpu().numpy()

        # search, rank, and print
        #scores = np.dot(vecs.T, qvecs)
        #ranks = np.argsort(-scores, axis=0)
        
        cscores = np.dot(cvecs.T, cqvecs)
        cranks = np.argsort(-cscores, axis=0)

        escores=np.dot(evecs.T,eqvecs)
        eranks=np.argsort(-escores,axis=0)

        iscores=np.dot(ivecs.T,iqvecs)
        iranks=np.argsort(-iscores,axis=0)

        '''cscores = torch.mm(cpoolvecs.t(), cqoolvecs)
        cranks = torch.argsort(-cscores, axis=0).cpu().numpy()

        escores = torch.mm(epoolvecs.t(), eqoolvecs)
        eranks = torch.argsort(-escores, axis=0).cpu().numpy()

        iscores = torch.mm(ipoolvecs.t(), iqoolvecs)
        iranks = torch.argsort(-iscores, axis=0).cpu().numpy()'''

        rrank=[cranks,eranks,iranks]

        for index,ranks in enumerate(rrank):
            if index==0:
                print('classification................')
            elif index==1:
                print('extractor.....................')
            else:
                print('intersect......................')

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

            res[mmap[index]]={'MAP':map,'MRR':mrr}

        return res

if __name__=='__main__':

    global args
    args = parser.parse_args()
    cuda_gpu = torch.cuda.is_available()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    result={}
    ip=get_host_ip()
    result['config']={'ip':ip,'autoaugment':args.autoaugment,'LR':args.LR,'finetunedir':args.train_dir}

    for i in range(14,25):
        pklword = args.train_dir.split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (i)
        path = args.train_dir.replace(pklword, newpkl)
        print(path)
        args.train_dir= path

        if not os.path.exists(path):
            continue


        res = testmodel(args, cuda_gpu)

        result[str(i)]={'path':path,'result':res}


    json.dump(result,open(now+'eyem1.json','w'))
