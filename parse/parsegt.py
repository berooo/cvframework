import os
import json
queryList={}

def parseoxford(rootimg,path,topath):
  g = os.walk(path)
  for path, dir_list, file_list in g:
    for file_name in file_list:
      filepath = os.path.join(path, file_name)
      strips = file_name.split('_')

      suffix = strips[-1]
      prefix = file_name.replace("_" + suffix, "")

      if prefix not in queryList:
        queryList[prefix] = {'queryimgid': '', 'ok': [], 'junk': [], 'good': [],'boxes':[]}
      if suffix == 'query.txt':
        file = open(filepath)
        for line in file:
          imgName = line.split(' ')[0]
          boxes=[float(s) for s in line.split(' ')[1:]]
          pre = imgName.split('_')[0]
          imgName = imgName.replace(pre + '_', '')
          queryList[prefix]['queryimgid'] = rootimg + imgName + '.jpg'
          queryList[prefix]['boxes']=boxes
        file.close()
      elif suffix == 'ok.txt':
        file = open(filepath)
        for line in file:
          queryList[prefix]['ok'].append(rootimg + line.strip() + '.jpg')
        file.close()
      elif suffix == 'junk.txt':
        file = open(filepath)
        for line in file:
          queryList[prefix]['junk'].append(rootimg + line.strip() + '.jpg')
        file.close()
      else:
        file = open(filepath)
        for line in file:
          queryList[prefix]['good'].append(rootimg + line.strip() + '.jpg')
        file.close()

  json.dump(queryList, open(topath, 'w'))


def parseparis(rootimg,path,topath):
  g = os.walk(path)
  for path, dir_list, file_list in g:
    for file_name in file_list:
      filepath = os.path.join(path, file_name)
      strips = file_name.split('_')
      suffix = strips[-1]
      prefix = file_name.replace("_" + suffix, "")
      if prefix not in queryList:
        queryList[prefix] = {'queryimgid': '', 'ok': [], 'junk': [], 'good': [],'boxes':[]}
      if suffix == 'query.txt':
        file = open(filepath)
        for line in file:
          imgName = line.split(' ')[0]
          boxes=[float(s) for s in line.split(' ')[1:]]
          pre = imgName.split('_')[1]
          imgName = pre+'/'+imgName
          queryList[prefix]['queryimgid'] = rootimg + imgName + '.jpg'
          queryList[prefix]['boxes']=boxes
        file.close()
      elif suffix == 'ok.txt':
        file = open(filepath)
        for line in file:
          imgName = line.strip()
          pre = imgName.split('_')[1]
          imgName = pre + '/' + imgName
          queryList[prefix]['ok'].append(rootimg + imgName + '.jpg')
        file.close()
      elif suffix == 'junk.txt':
        file = open(filepath)
        for line in file:
          imgName = line.strip()
          pre = imgName.split('_')[1]
          imgName = pre + '/' + imgName
          queryList[prefix]['junk'].append(rootimg + imgName + '.jpg')
        file.close()
      else:
        file = open(filepath)
        for line in file:
          imgName = line.strip()
          pre = imgName.split('_')[1]
          imgName = pre + '/' + imgName
          queryList[prefix]['good'].append(rootimg + imgName + '.jpg')
        file.close()
  json.dump(queryList, open(topath, 'w'))

def parsegt(path,topath,mode='oxford'):
  if mode=='oxford':
    rootimg='/mnt/sdb/shibaorong/data/oxford5k/train/'
    parseoxford(rootimg,path,topath)
  elif mode=='paris':
    rootimg='/mnt/sdb/shibaorong/data/paris6k/train/'
    parseparis(rootimg,path,topath)


if __name__=='__main__':
  root='/mnt/sdb/shibaorong/data/paris6k/query'
  topath='/home/shibaorong/modelTorch/out/parisquery.json'
  parsegt(root,topath,mode='paris')