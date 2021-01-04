import os
import sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import random as rd
import json
import csv

def parseToJson(dataDir,totrainFile,toTestFile,propotion,mode=0):

    saveList=[]
    i=0
    count=0
    label_dict={}
    for root,dirs,files in os.walk(dataDir):
        '''if root.split('/')[-1]=='general':
            continue'''
        for each in files:

            path=os.path.join(root,each)

            if mode==0:
                label=root.split('/')[-1]
            elif mode==1:
                label=each.replace('_'+each.split('_')[-1],'')
                if label=='oxford':
                    continue
            elif mode==2:
                label=each.split('.')[0]
            if label in label_dict:
                label_id=label_dict[label]
            else:
                label_id=count
                label_dict[label]=count
                count+=1
            i += 1
            ID = i
            record={'ID':ID,'label_id':label_id,'label':label,'filenames':path}
            saveList.append(record)
    print(label_dict)
    rd.shuffle(saveList)
    length = len(saveList)
    len_train = int(length * propotion)
    json.dump(saveList[0:len_train], open(totrainFile, 'w'))
    json.dump(saveList[len_train:], open(toTestFile, 'w'))

def parseeye(root,toTrainFile,toTestFile,propotion):
    dataDir=root+'/eyetrain/'
    csvdir=root+'/eyetrain.csv'
    reader = csv.reader(open(csvdir, 'r'))
    saveList = []
    for index, item in enumerate(reader):
        if reader.line_num == 1:
            continue
        name=item[2]
        label=int(item[3])
        ID=item[0]
        path=dataDir+name+'.jpeg'
        record = {'ID': ID, 'label_id': label, 'label': label, 'filenames': path}
        saveList.append(record)
    rd.shuffle(saveList)
    length = len(saveList)
    len_train = int(length * propotion)
    json.dump(saveList[0:len_train], open(toTrainFile, 'w'))
    json.dump(saveList[len_train:], open(toTestFile, 'w'))

def parseholidays(root,toTrainFile,toTestFile):
    savetrainList = []
    savetestList=[]
    for dirpath,dirname,filename in os.walk(root):
        for f in filename:
            label=int(f[1:4])
            path=os.path.join(dirpath,f)
            ID=f.split('.')[0]
            index=int(f[4:].split('.')[0])
            record = {'ID': ID, 'label_id': label, 'label': label, 'filenames': path}
            if index==0:
                savetestList.append(record)
            savetrainList.append(record)
    json.dump(savetrainList,open(toTrainFile,'w'))
    json.dump(savetestList,open(toTestFile,'w'))

if __name__=='__main__':
    '''dataDir='/mnt/sdb/shibaorong/data/pathological'
    mode=0
    toTrainFile='/home/shibaorong/modelTorch/out/breasttrain.json'
    toTestFile='/home/shibaorong/modelTorch/out/breasttest.json'
    propotion=1.0
    parseToJson(dataDir,toTrainFile,toTestFile,propotion,mode)'''

    dataDir='/mnt/sdb/shibaorong/data/holidays'
    toTrainFile = '/home/shibaorong/modelTorch/out/holidaystrain.json'
    toTestFile = '/home/shibaorong/modelTorch/out/holidaystest.json'
    parseholidays(dataDir,toTrainFile,toTestFile)


