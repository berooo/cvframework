import json

def cattriplet(fromfile):
  label_dict = {}
  data = json.load(open(fromfile))
  for d in data:
    for j in d:
      label = j['label_id']
      if label in label_dict:
        label_dict[label] += 1
      else:
        label_dict[label] = 1
  label_dict = sorted(label_dict.items(), key=lambda item: item[0])
  print(label_dict)

def catpair(fromfile):
  label_dict = {}
  data = json.load(open(fromfile))
  for d in data:
    for j in d:
      label = j['label_id']
      if label in label_dict:
        label_dict[label] += 1
      else:
        label_dict[label] = 1
  label_dict = sorted(label_dict.items(), key=lambda item: item[0])
  print(label_dict)

def catDatasets(fromfile):
  label_dict = {}
  data = json.load(open(fromfile))
  for d in data:
    label = d['label_id']
    if label in label_dict:
      label_dict[label] += 1
    else:
      label_dict[label] = 1
  label_dict=sorted(label_dict.items(),key=lambda item:item[0])
  print(label_dict)

if __name__=='__main__':
  file1='/home/shibaorong/modelTorch/out/train20200304.json'
  catDatasets(file1)
  file2 = '/home/shibaorong/modelTorch/out/test20200304.json'
  catDatasets(file2)
  file3='/home/shibaorong/modelTorch/out/trainpair.json'
  catpair(file3)
  file4='/home/shibaorong/modelTorch/out/traintriplet.json'
  cattriplet(file4)