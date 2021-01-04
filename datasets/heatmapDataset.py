import cv2
import requests

import numpy as np
import torchvision.transforms as transforms
import json

from torch.utils.data import Dataset
from util.util import *

class heatmapDataset(Dataset):
  def __init__(self, path, height=224, width=224,autoaugment=False,  transform=transforms.Compose([transforms.ToTensor()])):
    #self.items=[]

    self.items=json.load(open(path))
    '''for d in data.keys():
      item = data[d]
      for i in item:
        self.items.append(i)'''
    self.autoaugment = autoaugment
    self.transform = transform
    self.bbox_crop=BBoxCrop()
    self.rescale224square = Rescale((height, width))

  def checkLandmark(self,image,landmark_vis,landmark_in_pic,landmark_pos):
    h,w=image.shape[:2]
    landmark_vis=landmark_vis.copy()
    landmark_in_pic=landmark_in_pic.copy()
    landmark_pos=landmark_pos.copy()
    for i,vis in enumerate(landmark_vis):
      if (landmark_pos[i,0]<0 or (landmark_pos[i,0]>=w) or (landmark_pos[i,1]<0) or (landmark_pos[i,1]>=h)):
        landmark_vis[i]=0
        landmark_in_pic[i]=0

    for i,in_pic in enumerate(landmark_in_pic):
      if in_pic==0:
        landmark_pos[i,:]=0

    return landmark_vis,landmark_in_pic,landmark_pos

  def landmarkNormalize(self,image,landmark_pos):
    h,w=image.shape[:2]
    landmark_pos=landmark_pos/[float(w),float(h)]
    return landmark_pos

  def gaussian_map(self, image_w, image_h, center_x, center_y, R):
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

  def gen_landmark_map(self,image_w,image_h,landmark_in_pic,landmark_pos,R):
    ret=[]
    for i in range(len(landmark_in_pic)):
      if landmark_in_pic[i]==0:
        ret.append(np.zeros((image_w,image_h)))
      else:
        channel_map=self.gaussian_map(image_w,image_h,landmark_pos[i][0],landmark_pos[i][1],R)
        ret.append(channel_map.reshape((image_w,image_h)))
    return np.stack(ret,axis=0).astype(np.float32)

  def __getitem__(self, index):
    item=self.items[index]
    url=item['url']
    while True:
      try:
        resp = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      except Exception as e:
        print(e)
        continue
      break
    orig_image_size=image.shape
    ret = {}
    ret['url']=url

    landmark_vis=np.array(list(map(int,item['landmarks'][2::3])))
    numkeypoints=int(len(item['landmarks'])/3)
    landmark_pic=[1]*numkeypoints

    for i in range(len(landmark_vis)):
      if landmark_vis[i]==0:
        landmark_pic[i]=0

    landmark_pos_x=np.array(list(map(int,item['landmarks'][0::3]))).reshape(-1,1)
    landmark_pos_y=np.array(list(map(int,item['landmarks'][1::3]))).reshape(-1,1)
    landmark_pos=np.concatenate([landmark_pos_x,landmark_pos_y],axis=1)
    ret['orglandmarkpos']=landmark_pos
    bbox=item['bounding_box']
    image,landmark_pos=self.bbox_crop(image,landmark_pos,bbox[0],bbox[1],bbox[2],bbox[3])
    crop_image_size=image.shape[:2]
    image,landmark_pos=self.rescale224square(image,landmark_pos)

    landmark_vis,landmark_pic,landmark_pos=self.checkLandmark(image,landmark_vis,landmark_pic,landmark_pos)
    landmark_pos=landmark_pos.astype(np.float32)
    landmark_pos_normalized=self.landmarkNormalize(image,landmark_pos).astype(np.float32)

    image=self.transform(image)



    ret['image']=image
    ret['landmark_vis']=landmark_vis
    ret['landmark_in_pic']=np.array(landmark_pic)
    ret['landmark_pos']=landmark_pos
    ret['landmark_pos_normalized']=landmark_pos_normalized
    image_h,image_w=image.size()[1:]
    R=numkeypoints
    ret['landmark_map']=self.gen_landmark_map(image_w,image_h,landmark_pic,landmark_pos,R)
    ret['landmark_map224']=self.gen_landmark_map(image_w,image_h,landmark_pic,landmark_pos,R)
    ret['org_image_size']=np.array(orig_image_size)
    ret['cropped_image_size']=np.array(crop_image_size)
    ret['image_id']=self.items[index]['image_id']
    ret['bounding_box']=item['bounding_box']
    ret['category_id']=item['category_id']
    ret['style']=item['style']
    ret['segmentation']=item['segmentation']
    ret['pair_id']=item['pair_id']
    ret['image_id']=item['image_id']
    return ret

  def __len__(self):
    return len(self.items)

