#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: calMap.py
@time: 2019/9/1 19:19
@desc:
'''

import numpy as np
import time
import os
from util.distances import *

class eval(object):
    def __init__(self,id_lists,logits_lists,relu_ip1_list,label_lists,featurelib):
        self._ids=id_lists
        self._logits=logits_lists
        self._labels=label_lists
        self._featurelib=featurelib
        self._relu_ip1_list=relu_ip1_list

    #extract featurelib
    def extract(self):
        self._imagefeature=[]
        self._additionalfeature=[]
        self._truelabel=[]
        self._filenames=[]

        for image_feature in self._featurelib:
            self._imagefeature.append(image_feature[1][1])
            self._additionalfeature.append(image_feature[2][1])
            self._truelabel.append(image_feature[3][1])
            self._filenames.append(image_feature[0][1])

        self._indexes=[i for i in range(len(self._filenames))]

    def caltop1(self):
        pass

    def caltop5(self):
        pass

    def calAP(self,index):
        logs=[]
        logit=self._logits[index]
        logs.append(logit)
        
        reluip=self._relu_ip1_list[index]

        
        lab=self._labels[index]
        dist=[]
        dist2=[]

        for image_featurre in self._imagefeature:
            dis=getHammingDist(logs,image_featurre)
            dist.append(dis)
        
        for additional_feature in self._additionalfeature:
            diss=getSecondDist(reluip,additional_feature)
            dist2.append(diss)
            
        tump=list(zip(self._indexes,dist,dist2))
        tump=sorted(tump,key=lambda x:(x[1],x[2]))
        res_index=list(zip(*tump))[0]
        count=0
        top1=0
        top5=0
        total_precision=0.0
        for idx,i in enumerate(res_index):
            truelab=int(self._truelabel[i])
            if lab==truelab:
                count+=1
                if idx==0:
                    top1=1
                if idx<5:
                    top5=1

                total_precision+=count/(idx+1)
        
        if top5==0:
          print(self._ids[index])
          
          
        return top1,top5,total_precision/count

    def calMAP(self):
        totalfn=len(self._ids)
        totalmap=0.0
        t1=0
        t5=0
        for i in range(totalfn):
            top1,top5,mapp=self.calAP(i)
            totalmap+=mapp
            t1+=top1
            t5+=top5
            if i % 10 == 0 or i == totalfn - 1:
                print('Step %d, %.3f%% calculated. top1: %d. top5: %d.'%(i+1,(i + 1) / totalfn * 100,t1,t5))

        return t1/totalfn,t5/totalfn,totalmap/totalfn
