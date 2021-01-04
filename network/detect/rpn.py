from torch import nn
import torch
import numpy as np
import six
import torch.nn.functional as F
from network.detect.utils.creator_tool import ProposalTargetCreator,proposalCreator
def normal_init(m,mean,stddev,truncated=False):
    """
        weight initalizer: truncated normal and random normal.
        """
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def generate_anchor_base(base_size=16,ratios=[0.5,1,2],anchor_scales=[8,16,32]):

    py=base_size/2
    px=base_size/2

    anchor_base=np.zeros((len(ratios)*len(anchor_scales),4),dtype=np.float32)

    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index=i*len(anchor_scales)+j
            anchor_base[index,0]=py-h/2
            anchor_base[index,1]=px-w/2
            anchor_base[index,2]=py+h/2
            anchor_base[index,3]=px+w/2

    return anchor_base



def _enumberate_shifted_anchor_torch(anchor_base,feat_stride,height,width):
    #将所有的锚点坐标存储并且将数组转为torch.cudatensor
    shift_y=torch.arange(0,height*feat_stride,feat_stride)
    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    print(shift_y)

    A=anchor_base.shape[0]
    K=shift.shape[0]
    anchor=anchor_base.reshape((1,A,4))+shift.reshape((1,K,4)).transpose((1,0,2))
    anchor=anchor.reshape((K*A,4)).astype(np.float32)

    return anchor

class RPN(nn.Module):
    def __init__(self,in_channels=512,mid_channels=512,proposal_creator_params=dict(),**kwargs):
        super(RPN,self).__init__()
        self.anchor_base=generate_anchor_base(ratios=kwargs['ratios'],
                                              anchor_scales=kwargs['anchor_scales'])
        self.feat_stride=kwargs['feat_stride']
        n_anchor=self.anchor_base.shape[0]
        self.proposal_layer=proposalCreator(self,**proposal_creator_params)
        self.conv1=nn.Conv2d(in_channels,mid_channels,3,1,1)
        self.score=nn.Conv2d(mid_channels,n_anchor*2,1,1,0)
        self.loc=nn.Conv2d(mid_channels,n_anchor*4,1,1,0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self,x,img_size,scale=1.):
        n,_,ww,hh=x.shape()

        anchor=_enumberate_shifted_anchor_torch(
            np.array(self.anchor_base),self.feat_stride,hh,ww
        )
        n_anchor=anchor.shape[0]
        h=F.relu(self.conv1(x))
        rpn_locs=self.loc(h)
        rpn_locs=rpn_locs.permute(0,2,3,1).contiguous().view(n,-1,4)
        rpn_scores=self.score(h)
        rpn_scores=rpn_scores.permute(0,2,3,1).contiguous()
        rpn_softmax_scores=F.softmax(rpn_scores.view(n,hh,ww,n_anchor,2),dim=4)
        rpn_fg_scores=rpn_softmax_scores[:,:,:,:,1].contiguous()
        rpn_fg_scores=rpn_fg_scores.view(n,-1)
        rpn_scores=rpn_scores.view(n,-1,2)

        rois=list()
        roi_indices=list()
        for i in range(n):
            roi=self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor,img_size,
                scale=scale
            )
            batch_index=i*np.ones((len(roi),),dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois=np.concatenate(rois,axis=0)
        roi_indices=np.concatenate(roi_indices,axis=0)

        return rpn_locs,rpn_scores,rois,roi_indices,anchor



if __name__=='__main__':
    anchor=generate_anchor_base()
    anchor=_enumberate_shifted_anchor_torch(anchor,feat_stride=16,height=37,width=50)
    print(anchor)
