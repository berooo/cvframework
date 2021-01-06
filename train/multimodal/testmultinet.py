# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/5/30 19:04
"""
#拿出所有C列表
#拿出所有P列表
#rank
#查看rank的名字，排序
import sys



sys.path.insert(0, '../../')
import os
import random
import sys
from graph import builGraph
from scipy.io import savemat,loadmat
from PIL import Image
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import argparse
import os
import torch
from extract.multinet_utils import MultinetExtraction
from network.multimodal.multinet import multi_net
from datasets.imageListDateset import ImagesFromList
from datasets.CartoonDataset import preclsDataset
import numpy as np
import torchvision.transforms as transforms

imgdir='../../datasets/data/test'
query_path='../../datasets/data/gd/query.txt'
gallery_path='../../datasets/data/gd/gallery.txt'
NUM_EMBEDDING_DIMENSIONS=512
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('--batch_size',default=1,
                    help='destination where trained network should be saved')
parser.add_argument('--data_dir',default='../../datasets/data/train',
                    help='destination where trained network should be saved')
parser.add_argument('--train_dir',default='../../out/normalcls/parameter_180.pkl',
                    help='destination where trained network should be saved')
parser.add_argument('--autoaugment',default=False,
                    help='destination where trained network should be saved')
parser.add_argument('--backbone',default='vgg16',
                    help='destination where trained network should be saved')
parser.add_argument('--classnum',default=124,
                    help='destination where trained network should be saved')
parser.add_argument('--optimizer',default='adam',
                    help='destination where trained network should be saved')
parser.add_argument('--LR',default=0.01,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu',default=[0,1],
                    help='destination where trained network should be saved')
parser.add_argument('--maxepoch',default=2000,
                    help='destination where trained network should be saved')

parser.add_argument('--height',default=224,
                    help='destination where trained network should be saved')
parser.add_argument('--width',default=224,
                    help='destination where trained network should be saved')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

args = parser.parse_args()

def setup_model():
    model=builGraph.getModel('vgg16', 124, [0,1],
                                 'retrieval', cuda_gpu=True,pretrained=True)
    #model=multi_net(modelName=args.backbone)
    #model=MultinetExtraction(modelName=args.backbone)
    '''if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()'''
    load_checkpoint(args.train_dir,model)
    model.eval()
    return model

def preprocess(img):
    img=Image.open(img).convert('RGB')
    img=transform(img)
    img=img.unsqueeze(0)
    return img

def get_img_name(path,imgdir):
    imgpaths=[]
    for line in open(path):
        imgpaths.append(os.path.join(imgdir,line.strip()+'.jpg'))
    return imgpaths

def load_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def generate_embedding_single(model,imgpaths):

    num_embeddings=len(imgpaths)
    embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))
    for i, image_path in enumerate(imgpaths):

        input_data = preprocess(image_path)
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        feature=model(input_data)

        embeddings[i, :] = feature.cpu().detach().numpy()
        print(str(i) + ',' + str(len(imgpaths)))

    return embeddings

def generate_embedding(model,imgpaths,mode):

    num_embeddings=len(imgpaths)
    embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))
    mf_embeddings=[]
    for i, image_path in enumerate(imgpaths):

        input_data = preprocess(image_path)
        if torch.cuda.is_available():
            input_data = input_data.cuda()

        mf,feature=model(input_data,mode)
        mf_embeddings.append(mf.cpu().detach().numpy().reshape(-1))
        embeddings[i, :] = feature.cpu().detach().numpy()
        print(str(i) + ',' + str(len(imgpaths)))
    mf_embeddings=np.asarray(mf_embeddings)
    return mf_embeddings,embeddings

def testmultinet(C,P):

    model = setup_model()
    imgpathc=[i[1] for i in C]
    imgpathp=[i[1] for i in P]
    query_embeddings = generate_embedding_single(model, imgpathc)
    gallery_embeddings = generate_embedding_single(model, imgpathp)
    savemat('cartoontest.mat', {'C': query_embeddings, 'P': gallery_embeddings})
    '''features = loadmat('cartoontest.mat')
    query_embeddings = features['C']
    gallery_embeddings = features['P']'''
    scores = np.dot(gallery_embeddings, query_embeddings.T)
    ranks = np.argsort(-scores, axis=0)
    '''Q=query_embeddings
    X=gallery_embeddings
    dis1 = np.zeros([X.shape[0], Q.shape[0]])
    for j in range(Q.shape[0]):
        d = (X - np.reshape(Q[j, :], (1, Q.shape[1]))) ** 2
        disj = np.sum(d, axis=1)
        dis1[:, j] = disj'''

    '''Q = mf_q_embeddings
    X = mf_g_embeddings
    dis2 = np.zeros([X.shape[0], Q.shape[0]])
    for j in range(Q.shape[0]):
        d = (X - np.reshape(Q[j, :], (1, Q.shape[1]))) ** 2
        disj = np.sum(d, axis=1)
        dis2[:, j] = disj'''

    '''dis=dis1
    ranks = np.argsort(dis, axis=0)'''
    h,w=ranks.shape
    np.save("rankstrain.npy", ranks)
    count=0
    for i in range(w):
        qres=[P[ranks[j,i]] for j in range(30)]
        if qres[0][0]==C[i][0]:
           count+=1
        else:
            print(i)
    print('acc:{}'.format(float(count)/w))



def get_query_and_gallery(data_path):

    C = []
    P = []

    for index, name in enumerate(sorted(os.listdir(data_path))):
        imgroot = os.path.join(data_path, name)
        for imgname in os.listdir(imgroot):
            imgpath = os.path.join(imgroot, imgname)
            if imgname[0] == 'C':
                C.append((index, imgpath))
            else:
                P.append((index, imgpath))

    return C,P

def main1():
    C,P=get_query_and_gallery(data_path=args.data_dir)
    testmultinet(C,P)

def testbranchc(gallerys,querys):
    model = setup_model()
    query_embeddings = generate_embedding(model, querys, mode='c')
    gallery_embeddings = generate_embedding(model, gallerys, mode='c')
    scores = np.dot(gallery_embeddings, query_embeddings.T)
    ranks = np.argsort(-scores, axis=0)
    np.save("ranksbeanchc.npy", ranks)
    '''count = 0
    h, w = ranks.shape
    for i in range(w):
        qres = [C[ranks[j, i]] for j in range(10)]
        a=[k[0] for k in qres].count(C[i][0])
        if qres[0][0] == C[i][0]:
            count += a
        else:
            print(i)
    print('acc:{}'.format(float(count) / (10*w)))'''

def main():
    gallerys = get_img_name(query_path, imgdir)
    querys=sorted(gallerys)[:50]
    testbranchc(gallerys,querys)

def main3():
    C, P = get_query_and_gallery(data_path=args.data_dir)
    gallerys = [i[1] for i in C]
    querys = gallerys[:50]
    testbranchc(gallerys,querys,C)

def main_main():
    querys=get_img_name(query_path,imgdir)
    gallerys=get_img_name(gallery_path,imgdir)
    model=setup_model()
    '''model=builGraph.getModel('vgg16', 124, [0,1],
                                 'retrieval', cuda_gpu=True,pretrained=True)'''
    query_embeddings=generate_embedding_single(model,querys)
    gallery_embeddings=generate_embedding_single(model,gallerys)
    scores = np.dot(gallery_embeddings, query_embeddings.T)
    ranks = np.argsort(-scores, axis=0)
    np.save("ranks.npy", ranks)

if __name__=='__main__':
    main1()