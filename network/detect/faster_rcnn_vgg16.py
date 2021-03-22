from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from .rpn import RPN,normal_init
from .faster_rcnn import FasterRCNN
from util import array_tool as at
from .config.detectvgg16config import opt
from .utils.bboxs_tools import ROIPool

def decom_vgg16():

    model=vgg16(not opt.load_path)
    features=list(model.features)[:30]
    classifier=model.classifier
    classifier=list(classifier)
    del classifier[6]

    if not opt.use_drop:
        del classifier[5]
        del classifier[2]

    classifier=nn.Sequential(*classifier)

    for layer in features[:10]:
        for p in layer.parameters():
            p.require_grad=False

    return nn.Sequential(*features),classifier

class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
        For descriptions on the interface of this model, please refer to
        :class:`model.faster_rcnn.FasterRCNN`.
        Args:
            n_fg_class (int): The number of classes excluding the background.
            ratios (list of floats): This is ratios of width to height of
                the anchors.
            anchor_scales (list of numbers): This is areas of anchors.
                Those areas will be the product of the square of an element in
                :obj:`anchor_scales` and the original area of the reference
                window.
        """
    feat_stride=16
    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5,1,2],
                 anchor_scales=[8,16,32]):
        extractor,classifier=decom_vgg16()
        kwargs={}
        kwargs['ratios']=ratios
        kwargs['anchor_scales']=anchor_scales
        kwargs['feat_stride']=self.feat_stride
        rpn=RPN(in_channels=512,mid_channels=512,**kwargs)
        head=VGG16roIHead(
            n_class=n_fg_class+1,
            roi_size=7,
            spatial_scale=(1./self.feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor,rpn,head
        )


class VGG16roIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
        This class is used as a head for Faster R-CNN.
        This outputs class-wise localizations and base based on feature
        maps in the given RoIs.

        Args:
            n_class (int): The number of classes possibly including the background.
            roi_size (int): Height and width of the feature maps after RoI-pooling.
            spatial_scale (float): Scale of the roi is resized.
            classifier (nn.Module): Two layer Linear ported from vgg16
        """
    def __init__(self,n_class,roi_size,spatial_scale,
                 classifier):
        super(VGG16roIHead,self).__init__()
        self.classifier=classifier
        self.cls_loc=nn.Linear(4096,n_class*4)
        self.score=nn.Linear(4096,n_class)

        normal_init(self.cls_loc,0,0.001)
        normal_init(self.score,0,0.01)

        self.n_class=n_class
        self.roi_size=roi_size
        self.spatial_scale=spatial_scale
        self.roi=ROIPool((self.roi_size,self.roi_size),self.spatial_scale)

    def forward(self,x,rois,roi_indices):
        roi_indices=at.totensor(roi_indices).float()
        rois=at.totensor(rois).float()
        indices_and_rois=t.cat([roi_indices[:,None],rois],dim=1)
        xy_indices_and_rois=indices_and_rois[:,[0,2,1,4,3]]
        indices_and_rois=xy_indices_and_rois.contiguous()
        pool=self.roi(x,indices_and_rois)
        pool=pool.view(pool.size(0),-1)
        fc7=self.classifier(pool)
        roi_cls_locs=self.cls_loc(fc7)
        roi_scores=self.scores(fc7)
        return roi_cls_locs,roi_scores



