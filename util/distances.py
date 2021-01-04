#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: distances.py
@time: 2019/9/1 19:20
@desc:
'''
import numpy as np

#Hamming distance
def  getHammingDist(code_a,code_b):
    dist=0
    for i in range(len(code_a[0])):
        if(code_a[0][i]!=code_b[i]):
            dist+=1
    return dist
    
#second distance
def getSecondDist(relu_ip1, feature):

    f_arr=[float(i) for i in feature.split(',')[0:-1]]
    f_arr=np.asarray(f_arr)

    dis=np.sum(np.square(f_arr-relu_ip1))
    
    return dis