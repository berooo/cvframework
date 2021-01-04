import torch
import sys
import numpy as np
import json
import cv2
import util.util as tool
import os
import traceback
from  util.util import *
import util.distances as distance
from google.protobuf import json_format
from graph import builGraph
import heapq
import torchvision.transforms as transforms
from PIL import Image
from protos.model_pb2 import ModelConfig
from protos.train_pb2 import TrainConfig

feature_lib='features/oxford_logits_features.json'
modelConfig='config/retrieval/modeldshNet.config'
trainConfig='config/retrieval/traindshNet.config'
width=224
height=224

def getFeatureLib():
  with open(feature_lib, 'r') as f:
    featurelib_dict = json.loads(f.read())
    return featurelib_dict

def getQfeature(imgpath,modelconfig,trainconfig):
  img = Image.open(imgpath).convert('RGB')
  img = img.resize((modelconfig.width, modelconfig.height))
  img= np.array(img)
  transform = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  img=transform(img)
  img=torch.unsqueeze(img,dim=0)
  cuda_gpu = torch.cuda.is_available()

  mymodel = builGraph.getModel(modelconfig.backbone, modelconfig.num_classes, trainconfig.gpus,
                               modelconfig.modeltype, cuda_gpu=cuda_gpu)

  if os.path.exists(trainconfig.train_dir):
    checkpoint = torch.load(trainconfig.train_dir)
    mymodel.load_state_dict(checkpoint['model_state_dict'])

  mymodel.eval()
  with torch.no_grad():
    if cuda_gpu:
      batch_x = img.cuda()
    batch_x = batch_x.float()
    out, features = mymodel(batch_x)
    reluip = features['ip1'].cpu().numpy()
    logits = out.cpu().numpy()
    binarvalues = toBinaryString(logits)
  return binarvalues, reluip

def query(imgpath, output, rank,modelconfig,trainconfig):
  qfeature, relu_ip1 = getQfeature(imgpath,modelconfig,trainconfig)

  print(qfeature)
  featurelib = getFeatureLib()

  dist = []
  dist2 = []
  names = []
  for image_feature in featurelib:
    dis = distance.getHammingDist(qfeature, image_feature[1][1])
    dist.append(dis)
    diss = distance.getSecondDist(relu_ip1, image_feature[2][1])
    dist2.append(diss)
    names.append(image_feature[0][1])

  distlist = list(zip(names, dist, dist2))
  top_k = heapq.nsmallest(rank, distlist, key=lambda d: (d[1], d[2]))

  res = list(zip(*top_k))[0]

  with open(output, 'w') as f:
    f.write(json.dumps(res))
    print('Write result ok!')

def main():
  assert modelConfig != ''
  assert trainConfig != ''

  try:
    train_config = TrainConfig()
    model_config = ModelConfig()

    with open(modelConfig, 'r') as f:
      info = f.read()
      json_format.Parse(info, model_config)
    with open(trainConfig, 'r') as f:
      info = f.read()
      json_format.Parse(info, train_config)
  except:
    traceback.print_exc()
    print('error when parsing %s.', modelConfig)

  path = '/mnt/sdb/shibaorong/data/oxford5k/train/hertford_000027.jpg'
  query(path, 'n.txt', 10,model_config,train_config)


if __name__=='__main__':
  main()

