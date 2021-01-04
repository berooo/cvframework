import traceback
import torch
import os
import input
import tqdm
import time
import query
from google.protobuf import json_format
from protos.train_pb2 import TrainConfig
from protos.model_pb2 import ModelConfig
from input import *
from graph import builGraph,buildLoss
from torch.autograd import Variable
import torchvision.transforms as transforms
from util.util import to_Onehot
from option.calMap import *
import cv2
import numpy.matlib

trainConfig='config/glemtrain.config'
modelConfig='config/glemtest.config'

numpoints=25


def printdot(img, lm_pos, filename):
  zeroimg = img.copy().astype(np.uint8)
  point_size = 1
  point_color = (0, 0, 255)  # BGR
  thickness = 4
  for p in lm_pos:
    p = tuple(p)
    cv2.circle(zeroimg, p, point_size, point_color, thickness)

  cv2.imwrite(filename, zeroimg)


def eval(output):
  lm_pos_map = output
  numpoints, pred_h, pred_w = lm_pos_map.size()
  lm_pos_reshaped = lm_pos_map.reshape(numpoints, -1)
  lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=1).cpu().numpy(), (pred_h, pred_w))

  lm_pos = np.stack([lm_pos_x, lm_pos_y], axis=1)

  return lm_pos
def gaussian_map(image_w, image_h, center_x, center_y, R):
  Gauss_map = np.zeros((image_h, image_w))
  mask_x = np.matlib.repmat(center_x, image_h, image_w)
  mask_y = np.matlib.repmat(center_y, image_h, image_w)
  x1 = np.arange(image_w)
  x_map = np.matlib.repmat(x1, image_h, 1)
  y1 = np.arange(image_h)

  y_map = np.matlib.repmat(y1, image_w, 1)
  y_map = np.transpose(y_map)
  Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)
  Gauss_map = np.exp(-0.5 * Gauss_map / R)
  return Gauss_map


def gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R):
  ret = []
  for i in range(len(landmark_in_pic)):
    if landmark_in_pic[i] == 0:
      ret.append(np.zeros((image_w, image_h)))
    else:
      channel_map = gaussian_map(image_w, image_h, landmark_pos[i][0], landmark_pos[i][1], R)
      ret.append(channel_map.reshape((image_w, image_h)))
  return np.stack(ret, axis=0).astype(np.float32)

class Evaluator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.lm_vis_count_all=np.array([0.]*25)
        self.lm_dist_all=np.array([0.]*25)

    def add(self,output,sample,img):
        landmark_vis_count=sample['landmark_vis'].cpu().numpy().sum(axis=0)
        landmark_vis_float=torch.unsqueeze(sample['landmark_vis'].float(),dim=2)
        landmark_vis_float=torch.cat([landmark_vis_float,landmark_vis_float],dim=2).cpu().detach().numpy()

        lm_pos_map=output['lm_pos_map']
        batchsize,_,pred_h,pred_w=lm_pos_map.size()
        lm_pos_reshaped=lm_pos_map.reshape(batchsize,25,-1)
        lm_pos_y,lm_pos_x=np.unravel_index(torch.argmax(lm_pos_reshaped,dim=2).cpu().numpy(),(pred_h,pred_w))
        lm_pos=np.stack([lm_pos_x,lm_pos_y],axis=2)[0]
        zeroimg=np.zeros([224,224,3],dtype=np.uint8)
        zeroimg=img.copy().astype(np.uint8)
        point_size = 1
        point_color = (0, 0, 255)  # BGR
        thickness = 4
        for p in lm_pos:
          p=tuple(p)
          cv2.circle(zeroimg, p, point_size, point_color, thickness)

        cv2.imwrite('heatmap.jpg',zeroimg)

        lm_pos_output=np.stack([lm_pos_x/(pred_w-1),lm_pos_y/(pred_h-1)],axis=2)

        landmark_dist=np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float*lm_pos_output-landmark_vis_float*sample['landmark_pos_normalized'].cpu().numpy()
        ),axis=2)),axis=0)

        self.lm_vis_count_all+=landmark_vis_count
        self.lm_dist_all+=landmark_dist

    def evaluate(self):
        lm_dist=self.lm_dist_all/self.lm_vis_count_all
        lm_dist[np.isnan(lm_dist)] = 0
        lm_dist_all=lm_dist.mean()

        return {'lm_dist':lm_dist,
                'lm_dist_all':lm_dist_all}


def testLandmark(params, transform):
  mytraindata = heatmapDataset(path=params['valdata_dir'], height=params['height'], width=params['width'],
                               autoaugment=params['autoaugment'], transform=transform)
  mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)

  cuda_gpu = torch.cuda.is_available()
  mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                               params['model_type'], cuda_gpu=cuda_gpu)
  item=mytraindata.items[0]
  url=item['url']

  if os.path.exists(params['train_dir']):
    checkpoint = torch.load(params['train_dir'])
    mymodel.load_state_dict(checkpoint['model_state_dict'])

  mymodel.eval()
  test_step = len(mytrainloader)
  print('Evaluating.....')
  with torch.no_grad():
    evaluator = Evaluator()
    for i, sample in enumerate(mytrainloader):

      for key in sample:
        if isinstance(sample[key], list):
          continue
        sample[key] = sample[key].cuda().float()
      url=sample['url']
      landmarkpos = sample['orglandmarkpos']
      transfrompos = sample['landmark_pos']
      img=sample['image'][0].cpu().numpy()
      img=(img/2+0.5)*255
      img=img.astype(np.int)
      img=img.transpose((1,2,0))
      out = mymodel(sample)
      score_map=out['lm_pos_map']
      count=score_map.size(0)
      [x1s,y1s,x2s,y2s]=sample['bounding_box']
      for index in range(count):

        urli = url[index]
        while True:
          try:
            resp = requests.get(urli, stream=True).raw
            origimg = np.asarray(bytearray(resp.read()), dtype="uint8")
            origimg = cv2.imdecode(origimg, cv2.IMREAD_COLOR)
            origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)
          except Exception as e:
            print(e)
            continue
          break

        s = score_map[index, :, :, :]
        x1 = int(x1s[index].cpu().numpy())
        y1 = int(y1s[index].cpu().numpy())
        x2 = int(x2s[index].cpu().numpy())
        y2 = int(y2s[index].cpu().numpy())

        orh = y2 - y1
        orw = x2 - x1
        predict_pos = eval(s)
        ppos = predict_pos * [orh / 224, orw / 224]
        ppos += [x1, y1]
        ppos = [[int(i) for i in j] for j in ppos]

        printdot(origimg, ppos, 'origintest.jpg')

        true_pos = landmarkpos[index, :, :].int()
        printdot(origimg, true_pos, 'truetest.jpg')

        trans_pos = transfrompos[index, :, :].cpu().numpy()
        transpos = trans_pos * [orw / 224, orh / 224]
        transpos += [x1, y1]
        transpos = [[int(i) for i in j] for j in transpos]
        printdot(origimg, transpos, 'transtruetest.jpg')


        print('Val Step [{}/{}]'.format(i + 1, test_step))
    results = evaluator.evaluate()


def test(train_config, model_config):
  params = {}

  params['train_dir'] = train_config.train_dir
  params['data_dir'] = train_config.data_dir
  params['LR'] = train_config.initial_learning_rate
  params['train_method'] = train_config.optimizer
  params['class_num'] = model_config.num_classes
  params['valdata_dir'] = train_config.valdata_dir
  params['modelName'] = model_config.backbone
  params['height'] = model_config.height
  params['width'] = model_config.width
  params['BATCH_SIZE'] = train_config.batch_size
  params['maxepoch'] = train_config.max_epochs
  params['autoaugment'] = model_config.autoaugment
  params['Gpu'] = train_config.gpus
  params['model_type'] = model_config.modeltype
  params['featuredim'] = model_config.featuredim
  params['hashingbits'] = model_config.hashingbits

  transform = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


  testLandmark(params, transform)



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
    traceback.print_ex()
    print('error when parsing %s and %s.', trainConfig, modelConfig)

  test(train_config, model_config)


if __name__ == '__main__':
    import csv

    rows2 = ['abc1/ab1c', 'N']
    for n in range(10):
        f = open("ok.csv", 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(rows2)
        f.close()
  #main()