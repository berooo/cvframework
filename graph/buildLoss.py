import torch
import torch.nn as nn
import torch.nn.functional as F
import graph.function as LF
import numpy as np
from util.array_tool import scalar

class UWCSLoss(nn.Module):
  def __init__(self,clusterInfo,margin=0.7):
    super(UWCSLoss, self).__init__()
    iddict,center_mean,globalmean=clusterInfo
    self.iddict=iddict
    self.center_mean=center_mean
    self.global_mean=globalmean
    self.margin=margin

  def forward(self,output1, id1,output2,id2,target,size_average=True):
    distances = (output2 - output1).pow(2).sum(1).float()
    record1=[self.iddict[str(scalar(i))] for i in id1 ]
    record2=[self.iddict[str(scalar(i))] for i in id2 ]
    featuredis=[pow((record1[i]['feature']-record2[i]['feature']),2).sum() for i in range(len(record1))]
    featuredis=torch.from_numpy(np.asarray(featuredis)).cuda().float()
    label1=[i['label'] for i in record1]
    label2=[i['label'] for i in record2]
    center_mean=torch.from_numpy(np.asarray([self.center_mean[str(label)] for label in label1])).cuda().float()
    ctarget = np.zeros([len(label1), 1])
    for i in range(len(label1)):
      if label1[i] == label2[i]:
        ctarget[i, :] = 1
      else:
        ctarget[i, :] = 0
    ctarget=torch.from_numpy(ctarget).cuda().float()
    loss=0.5*distances*target*ctarget*center_mean[label1]/featuredis+\
      0.5*(1 + -1 * target)*(1 + -1 * target)*F.relu(self.margin-distances)*featuredis/self.global_mean

    return loss.mean() if size_average else loss.sum()

class ContrastiveLoss(nn.Module):
  """
  Contrastive loss
  Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
  """
  def __init__(self, margin=0.7):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, output1, output2, target, size_average=True):

    distances = (output2 - output1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target.float() * distances +
                    (1 + -1 * target).float() * F.relu(self.margin -distances) )
    return losses.mean() if size_average else losses.sum()


'''class TripletLoss(nn.Module):

  def __init__(self, margin=0.1):
    super(TripletLoss, self).__init__()
    self.margin = margin

  def forward(self, x, label):
    return LF.triplet_loss(x, label, margin=self.margin)

  def __repr__(self):
    return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'
'''
class TripletLoss(nn.Module):

  def __init__(self, margin=0.1):
    super(TripletLoss, self).__init__()
    self.margin = margin

  def forward(self, anchor, positive, negative, size_average=True):
    try:
      distance_positive=(anchor-positive).pow(2).sum(1)
      distance_negative=(anchor-negative).pow(2).sum(1)
      losses=F.relu(distance_positive-distance_negative+self.margin)
    except Exception as e:
      print(e)
    return losses.mean() if size_average else losses.sum()
'''class ContrastiveLoss(nn.Module):
  """
  Contrastive loss
  Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
  """
  def __init__(self, margin=0.7):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, output1, output2, target, size_average=True):

    distances = (output2 - output1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target.float() * distances +
                    (1 + -1 * target).float() * F.relu(self.margin -distances) )
    return losses.mean() if size_average else losses.sum()

class multiTrpletLoss(nn.Module):
  def __init__(self, margin=0.1):
    super(TripletLoss, self).__init__()
    self.margin = margin

  def forward(self, x, label):
    return getmultitripletLoss(x, label, margin=self.margin)

  def __repr__(self):
    return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

'''
class DshLoss(nn.Module):

  def __init__(self, margin):
    super(DshLoss, self).__init__()
    self.margin = margin
    self.alpha=0.01

  def forward(self, logits,label):
    batchsize=len(logits)
    w_label=torch.mm(label,label.t())
    r=(logits*logits).sum(1).view(1,-1)

    p_distance=r-2*torch.mm(logits,logits.t())+r.t()
    temp=w_label*p_distance+(1-w_label)*F.relu(self.margin-p_distance)
    regularizer=torch.abs(torch.abs(logits)-1).sum()

    d_loss=temp.sum()/(batchsize*(batchsize-1))+self.alpha*regularizer/batchsize
    return d_loss


def getloss(out, label, cuda_gpu=True, loss_type='crossentropy'):
  if loss_type == 'crossentropy':
    loss_func = torch.nn.CrossEntropyLoss()

  if cuda_gpu:
    loss_func=loss_func.cuda()
    out=out.cuda()
    label=label.cuda()
  loss = loss_func(out, label)

  return loss

def getDshloss(logits,label,hashingbits,alpha=0.01,cuda_gpu=True):
  m=hashingbits*2
  myloss=DshLoss(m)
  if cuda_gpu:
    myloss = myloss.cuda()
  return myloss(logits,label)

'''
def getSiameseloss(out1, out2, label, cuda_gpu=True):

  myloss=ContrastiveLoss(margin=1.0)

  if cuda_gpu:
    myloss = myloss.cuda()
  return myloss(out1,out2,label)


def getTripletloss(output,feature, plabel,nlabel,cuda_gpu=True):
  anchor,positive,negative=output[:,0],output[:,1],output[:,2]
  anchorf,positivef,negativef=feature[:,0],feature[:,1],feature[:,2]
  myloss = TripletLoss(margin=1.0)
  if cuda_gpu:
    myloss = myloss.cuda()
  tloss=myloss(anchorf, positivef, negativef)
  anchor=anchor.unsqueeze(0)
  positive=positive.unsqueeze(0)
  negative=negative.unsqueeze(0)
  plabel=plabel.unsqueeze(0)
  nlabel=nlabel.unsqueeze(0)
  labelossa=getloss(anchor,plabel)

  ttloss=labelossa
  return ttloss
'''
def getGLEMloss(sample,output,numkeypoints):

  batchsize,_,predh,predw=sample['image'].size()
  lm_size=int(output['lm_pos_map'].size(2))
  visibility=sample['landmark_vis']

  temp_list=[visibility.reshape(batchsize*numkeypoints,-1)]*lm_size*lm_size
  vis_mask=torch.cat(tuple(temp_list),1).float()
  lm_map_gt=sample['landmark_map%d' % lm_size].reshape(batchsize*numkeypoints,-1)
  lm_pos_map=output['lm_pos_map']
  lm_map_pred=lm_pos_map.reshape(batchsize*numkeypoints,-1)
  loss=torch.pow(vis_mask*(lm_map_pred-lm_map_gt),2).mean()

  return loss

def getmultitripletLoss(x, label, margin=0.1):
  dim = x.size(0)  # D
  nq = torch.sum(label.data == -1).item()  # number of tuples
  S = x.size(1) // nq  # number of images per tuple including query: 1+1+n

  xa = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, dim).permute(1, 0)
  xp = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, dim).permute(1, 0)
  xn = x[:, label.data == 0]

  dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
  dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

  return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0))