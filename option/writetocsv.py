import os
import numpy as np
from scipy.io import loadmat
import csv
#加载query.txt
#加载gallery.txt
#排序
#映射
#将文件名写到csv中

gallery_path='../datasets/data/gd/gallery.txt'
query_path='../datasets/data/gd/query.txt'
imgdir='../datasets/data/test'
tofile='submission.csv'

def get_img_name(path):
    ids=[]
    for line in open(path):
        ids.append(line.strip())
    return ids

def getresult(ranks,gallerys,querys):
    reslist=[]
    h,w=ranks.shape

    for i in range(w):
        qres=gallerys[ranks[0,i]]
        print(qres)
        print(i)
        reslist.append(qres)
    print(len(reslist))
    return reslist

def write_to_csv(res,tofile):

    with open(tofile,'w') as file:
        writer=csv.writer(file)
        for i in res:
          writer.writerow([i])


if __name__=='__main__':
    querys = get_img_name(query_path)
    gallerys = get_img_name(gallery_path)
    ranks=np.load('../train/multimodal/ranks.npy')
    res=getresult(ranks,gallerys,querys)
    write_to_csv(res,tofile)
