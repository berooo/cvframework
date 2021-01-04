import json
import random

def genPairs(jsonfile,tofile,mode='train',type='class'):
  positive_pairs=[]
  negative_pairs=[]
  data=json.load(open(jsonfile))
  length=len(data)
  p=0
  n=0
  while True:
    pair1,pair2=random.sample(data,2)
    if ([pair1,pair2] in positive_pairs) or ([pair1,pair2] in negative_pairs):
      continue

    if pair1['label_id']==pair2['label_id']:
      if len(positive_pairs)<length:
        positive_pairs.append([pair1,pair2])
        p+=1

    else:
      if len(negative_pairs)<length:
        negative_pairs.append([pair1,pair2])
        n+=1

    if len(positive_pairs)>=length and len(negative_pairs)>=length:
      break

  pairs=positive_pairs+negative_pairs
  random.shuffle(pairs)
  tofile+=mode+'pair.json'
  json.dump(pairs,open(tofile,'w'))

def genTriplet(jsonfile,tofile,mode='train',type='class'):
  tripletList=[]
  data=json.load(open(jsonfile))
  length=len(data)

  p=0

  while True:
    t1,t2,t3=random.sample(data,3)
    l1,l2,l3=t1['label_id'],t2['label_id'],t3['label_id']
    if l1!=l2 and l2!=l3 and l1!=l3:
      continue

    if l1==l2 and l2==l3:
      continue

    if l1==l2:
      tripletList.append([t1,t2,t3])
    elif l1==l3:
      tripletList.append([t1,t3,t2])
    else:
      tripletList.append([t2,t3,t1])
    p+=1
    if len(tripletList)>length:
      break
  tofile += mode + 'triplet.json'

  json.dump(tripletList, open(tofile, 'w'))
  print('-----------------------------------------------------------------')


if __name__=='__main__':
  train_file='/home/shibaorong/modelTorch/out/paris6ktrain.json'
  test_file='/home/shibaorong/modelTorch/out/pathologicaltest.json'
  tofile='/home/shibaorong/modelTorch/out/paris'
  genPairs(train_file,tofile)
  #genPairs(test_file,tofile,mode='test')

