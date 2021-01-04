import os
import numpy as np
from scipy.io import loadmat
import csv
#加载query.txt
#加载gallery.txt
#排序
#映射
#将文件名写到csv中

gallery_path='../data/cartoon/gallery.txt'
query_path='../data/cartoon/query.txt'
imgdir='/home/shibaorong/cartoon/extraction/data/cartoontest'
matpath='../extract/features/cartoon.mat'
tofile='submission.csv'

def get_img_name(path):
    ids=[]
    for line in open(path):
        ids.append(line.strip())
    return ids

def search(feature_path):
    features = loadmat(feature_path)
    Q = features['C']
    X = features['P']

    dis = np.zeros([X.shape[0], Q.shape[0]])

    for j in range(Q.shape[0]):
        d = (X - np.reshape(Q[j, :], (1, Q.shape[1]))) ** 2
        disj = np.sum(d, axis=1)
        dis[:, j] = disj
    ranks = np.argsort(dis, axis=0)
    '''sim = np.dot(X, Q.T)
    ranks = np.argsort(-sim, axis=0)'''
    np.save("ranks.npy", ranks)
    return ranks

def getresult(ranks,gallerys,querys):
    reslist=[]
    h,w=ranks.shape

    for i in range(w):
        qres=gallerys[ranks[0,i]]
        reslist.append(qres)

    return reslist

def write_to_csv(res,tofile):
    for line in res:
        with open(tofile,'a') as file:
            writer=csv.writer(file)
            writer.writerow([line])


if __name__=='__main__':
    querys = get_img_name(query_path)
    gallerys = get_img_name(gallery_path)
    ranks=search(matpath)
    res=getresult(ranks,gallerys,querys)
    write_to_csv(res,tofile)
