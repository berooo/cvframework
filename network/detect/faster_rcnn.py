from __future__ import absolute_import
from __future__ import division
import torch
import numpy as np
from util import array_tool as at
from network.detect.utils.bboxs_tools import loc2bbox, py_cpu_nms

from torch import nn


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)

    return new_f


class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.
        This is a base class for Faster R-CNN links supporting object detection
        API [#]_. The following three stages constitute Faster R-CNN.
        1. **Feature extraction**: Images are taken and their \
            feature maps are calculated.
        2. **Region Proposal Networks**: Given the feature maps calculated in \
            the previous stage, produce set of RoIs around objects.
        3. **Localization and Classification Heads**: Using feature maps that \
            belong to the proposed RoIs, classify the categories of the objects \
            in the RoIs and improve localizations.
        Each stage is carried out by one of the callable
        :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.
        There are two functions :meth:`predict` and :meth:`__call__` to conduct
        object detection.
        :meth:`predict` takes images and returns bounding boxes that are converted
        to image coordinates. This will be useful for a scenario when
        Faster R-CNN is treated as a black box function, for instance.
        :meth:`__call__` is provided for a scnerario when intermediate outputs
        are needed, for instance, for training and debugging.
        Links that support obejct detection API have method :meth:`predict` with
        the same interface. Please refer to :meth:`predict` for
        further details.
        .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
        Faster R-CNN: Towards Real-Time Object Detection with \
        Region Proposal Networks. NIPS 2015.
        Args:
            extractor (nn.Module): A module that takes a BCHW image
                array and returns feature maps.
            rpn (nn.Module): A module that has the same interface as
                :class:`model.region_proposal_network.RegionProposalNetwork`.
                Please refer to the documentation found there.
            head (nn.Module): A module that takes
                a BCHW variable, RoIs and batch indices for RoIs. This returns class
                dependent localization paramters and class scores.
            loc_normalize_mean (tuple of four floats): Mean values of
                localization estimates.
            loc_normalize_std (tupler of four floats): Standard deviation
                of localization estimates.
        """

    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    def use_preset(self, preset):
        """Use the given preset during prediction.
        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.
        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.
        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.
        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.
        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.
        Here are notations used.
        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.
        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.
        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.
        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.
            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.
        """
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs,rpn_scores,rois,roi_indices,anchor=\
            self.rpn(h,img_size,scale)
        roi_cls_locs,roi_scores=self.head(
            h,rois,roi_indices
        )

        return roi_cls_locs,roi_scores,rois,roi_indices

