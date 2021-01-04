from input import *
import traceback
import torch
import torch.nn as nn
import os
import util.util as tool
import time
from google.protobuf import json_format
from protos.train_pb2 import TrainConfig
from protos.model_pb2 import ModelConfig
from graph import builGraph
from torch.autograd import Variable
import torchvision.transforms as transforms

trainConfig='../config/retrieval/tripletdshtrain.config'
modelConfig='../config/retrieval/tripletdshmodel.config'
tofile='../features/paris_logits_features.json'

class FeatureExtractor(nn.Module):
  def __init__(self,submodel,extraclayers):
    super(FeatureExtractor, self).__init__()
    self.submodel=submodel
    self.extract_layers=extraclayers

  def getfeature(self,modules,x):
    output=[]
    for name,module in modules.features:
      print(name)
      if name=='module':
        output+=self.getfeature(module,x)
      else:
        x=module(x)
        if name in self.extract_layers:
          output.append(x)
    return output

  def forward(self,x):
    outputs=self.getfeature(self.submodel,x)
    return outputs

def buildfeaturelib(train_config, model_config):
  transform = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  mytraindata = myDataset(path=train_config.data_dir, height=model_config.height, width=model_config.width,
                          autoaugment=model_config.autoaugment, transform=transform)

  mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)

  cuda_gpu = torch.cuda.is_available()
  mymodel = builGraph.getModel(model_config.backbone, model_config.num_classes, train_config.gpus,
                               'onlinepair', cuda_gpu=cuda_gpu)

  if os.path.exists(train_config.train_dir):
    checkpoint = torch.load(train_config.train_dir)
    mymodel.load_state_dict(checkpoint['model_state_dict'])


  relu_ip1_list = []
  logits_list = []
  id_list = []
  label_list = []

  batch_idx=len(mytrainloader)//1
  #,batch_x,batch_y,batch_id
  mymodel.eval()
  with torch.no_grad():
    for index,data in enumerate(mytrainloader):
      batch_x,batch_y,batch_id=data
      if cuda_gpu:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
      batch_x = batch_x.float()
      #batch_x, batch_y = Variable(batch_x), Variable(batch_y)
      out,features = mymodel(batch_x)
      reluip=features['ip1'].cpu().numpy()
      logits=out.cpu().numpy()
      batch_y=batch_y.cpu().numpy()
      binarvalues=toBinaryString(logits)


      for binary in binarvalues:
        logits_list.append(binary)

      for unit in reluip:
        relu_ip1_list.append(unit)

      for id in batch_id:
        id_list.append(id)

      for label in batch_y:
        label_list.append(str(label))

      if index%10==0:

        tool.save(id_list,logits_list,relu_ip1_list,label_list,tofile)

      print('Step %d, %.3f%% extracted.' % (index, (index + 1)/ batch_idx * 100))


def main():
  assert trainConfig != ''
  assert modelConfig != ''

  try:
    train_config = TrainConfig()
    model_config = ModelConfig()

    with open(trainConfig, 'r') as f:
      info = f.read()
      json_format.Parse(info, train_config)

    with open(modelConfig, 'r') as f:
      info = f.read()
      json_format.Parse(info, model_config)
  except:
    traceback.print_exc()
    print('error when parsing %s and %s.', trainConfig, modelConfig)

  buildfeaturelib(train_config, model_config)

if __name__=='__main__':
  main()