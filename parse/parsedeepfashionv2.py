import json

short_sleeve_top=[]
long_sleeve_top=[]
short_sleeve_outwear=[]
long_leave_outwear=[]
vest=[]
sling=[]
shorts=[]
trousers=[]
skirt=[]
short_sleeve_dress=[]
long_sleeve_dress=[]
vest_dress=[]
sling_dress=[]

if __name__=='__main__':
  path='/home/shibaorong/modelTorch/data/deepfashion/deepfashionv2.train.data.json'
  toTrainFile = '/home/shibaorong/modelTorch/out/traindf'
  p=0
  data = json.load(open(path))
  for d in data.keys():
    item = data[d]
    for i in item:
      if i['category_id']==1:
        short_sleeve_top.append(i)
      elif i['category_id']==2:
        long_sleeve_top.append(i)
      elif i['category_id']==3:
        short_sleeve_outwear.append(i)
      elif i['category_id']==4:
        long_leave_outwear.append(i)
      elif i['category_id']==5:
        vest.append(i)
      elif i['category_id']==6:
        sling.append(i)
      elif i['category_id']==7:
        shorts.append(i)
      elif i['category_id']==8:
        trousers.append(i)
      elif i['category_id']==9:
        skirt.append(i)
      elif i['category_id']==10:
        short_sleeve_dress.append(i)
      elif i['category_id']==11:
        long_sleeve_dress.append(i)
      elif i['category_id']==12:
        vest_dress.append(i)
      elif i['category_id']==13:
        sling_dress.append(i)
      p+=1
      print(p)
  json.dump(short_sleeve_top, open(toTrainFile+'short_sleeve_top.json', 'w'))
  json.dump(long_sleeve_top, open(toTrainFile + 'long_sleeve_top.json', 'w'))
  json.dump(short_sleeve_outwear, open(toTrainFile + 'short_sleeve_outwear.json', 'w'))
  json.dump(long_leave_outwear, open(toTrainFile + 'long_leave_outwear.json', 'w'))
  json.dump(vest, open(toTrainFile + 'vest.json', 'w'))
  json.dump(sling, open(toTrainFile + 'sling.json', 'w'))
  json.dump(shorts, open(toTrainFile + 'shorts.json', 'w'))
  json.dump(trousers, open(toTrainFile + 'trousers.json', 'w'))
  json.dump(skirt, open(toTrainFile + 'skirt.json', 'w'))
  json.dump(short_sleeve_dress, open(toTrainFile + 'short_sleeve_dress.json', 'w'))
  json.dump(long_sleeve_dress, open(toTrainFile + 'long_sleeve_dress.json', 'w'))
  json.dump(vest_dress, open(toTrainFile + 'vest_dress.json', 'w'))
  json.dump(sling_dress, open(toTrainFile + 'sling_dress.json', 'w'))