# -*- coding: utf-8 -*-
"""
@author: shibaorong
@contact: diamond_br@163.com
@Created on: 2020/6/7 21:24
"""
from torch.utils.data import DataLoader
import numpy as np
import torch
from datasets.commonDataset import IDDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from util.array_tool import tonumpy
from util.array_tool import scalar

class kmeansCluster(object):
    def __init__(self,jsonpath,k,use_pca=False):
        dataset = IDDataset(jsonpath, autoaugment=False)
        self.dataloader=DataLoader(dataset=dataset,batch_size=1,shuffle=True)
        self.n_clusters=k
        self.use_pca = use_pca
        self.lendata = len(self.dataloader)
        self.ids=[]
        self.filenames=[]

    def clustering(self,mymodel,outdim):
        mymodel.eval()
        features = np.zeros([self.lendata, outdim])
        with torch.no_grad():
            for i,outputt in enumerate(self.dataloader):
                image, label, path, id=outputt

                out,feature = mymodel(image)
                #features[i, :] = tonumpy(feature['lastlinear'])
                features[i, :] = tonumpy(feature)
                '''feature=mymodel(image)
                features[i,:]=tonumpy(feature)'''
                self.ids.append(id)
                self.filenames.append(path)
        if self.use_pca:
            model2 = PCA(n_components=500, random_state=728)
            model2.fit(np.transpose(features))
            temp = np.transpose(model2.components_)
            self.images_new = temp
        else:
            self.images_new = features
        model = KMeans(n_clusters=self.n_clusters)
        s = model.fit(self.images_new)
        print('Kmeans-Clustering..........')
        iddict={}
        dict={}
        center_mean={}
        globalmean=0.
        centers = model.cluster_centers_
        labels = model.labels_
        l, f = centers.shape

        for i in range(l):
            dict[str(i)] = {'center': centers[i, :].tolist(), 'features': []}

        for i in range(self.lendata):
            dict[str(labels[i])]['features'].append(self.images_new[i].tolist())
            iddict[str(scalar(self.ids[i]))] = {}
            iddict[str(scalar(self.ids[i]))]['label'] = labels[i]
            iddict[str(scalar(self.ids[i]))]['feature'] = self.images_new[i]

        for d in dict:
            ft = np.asarray(dict[d]['features'])
            temp = pow(ft, 2).sum(1).reshape([ft.shape[0], -1])
            dis = temp - 2 * np.dot(ft, np.transpose(ft)) + np.transpose(temp)
            dis = np.around(dis, decimals=5).sum()
            n=ft.shape[0]
            if n!=1:
                dis/=n*(n-1)
            center_mean[d]=dis
            globalmean+=float(dis)/l

        return iddict,center_mean,globalmean
