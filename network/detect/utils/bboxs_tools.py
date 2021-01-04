import numpy as np
import numpy as xp
import torch.nn.functional as F
import torch
import six
from six import __init__

def bbox_iou(bbox_a,bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
        IoU is calculated as a ratio of area of the intersection
        and area of the union.
        This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
        inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
        same type.
        The output is same type as the type of the inputs.
        Args:
            bbox_a (array): An array whose shape is :math:`(N, 4)`.
                :math:`N` is the number of bounding boxes.
                The dtype should be :obj:`numpy.float32`.
            bbox_b (array): An array similar to :obj:`bbox_a`,
                whose shape is :math:`(K, 4)`.
                The dtype should be :obj:`numpy.float32`.
        Returns:
            array:
            An array whose shape is :math:`(N, K)`. \
            An element at index :math:`(n, k)` contains IoUs between \
            :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
            box in :obj:`bbox_b`.
        """
    if bbox_a.shape[1]!=4 or bbox_b.shape[1]!=4:
        raise IndexError

    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])

    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    return area_i / (area_a[:, None] + area_b - area_i)



def loc2bbox(src_bbox,loc):
    '''
    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            anchor
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.
    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.
    '''
    if src_bbox.shape[0]==0:
        return xp.zeros((0,4),dtype=loc.dtype)
    src_bbox=src_bbox.astype(src_bbox.dtype,copy=False)
    src_height=src_bbox[:,2]-src_bbox[:0]
    src_width=src_bbox[:,3]-src_bbox[:1]
    src_ctr_y=src_bbox[:,0]+0.5*src_height
    src_ctr_x=src_bbox[:,1]+0.5*src_width

    dy=loc[:,0::4]
    dx=loc[:,1::4]
    dh=loc[:,2::4]
    dw=loc[:,3::4]

    ctr_y=dy*src_height[:,xp.newaxis]+src_ctr_y[:,xp.newaxis]
    ctr_x=dx*src_width[:,xp.newaxis]+src_ctr_x[:,xp.newaxis]
    h=np.exp(dh)*src_height[:,np.newaxis]
    w=np.exp(dw)*src_width[:,np.newaxis]

    dst_bbox=np.zeros(loc.shape,dtype=loc.dtype)
    dst_bbox[:,0::4]=ctr_y-0.5*h
    dst_bbox[:,1::4]=ctr_x-0.5*w
    dst_bbox[:,2::4]=ctr_y+0.5*h
    dst_bbox[:,3::4]=ctr_x+0.5*w

    return dst_bbox

def py_cpu_nms(rois,scores,thresh):
    y1=rois[:,0]
    x1=rois[:,1]
    y2=rois[:,2]
    x2=rois[:,3]

    areas=(y2-y1+1)*(x2-x1+1)
    keep=[]
    index=scores.ravel().argsort()[::-1]

    while index.size>0:
        i=index[0]
        keep.append(i)

        x11=np.maximum(x1[i],x1[index[1:]])
        y11=np.maximum(y1[i],y1[index[1:]])
        x22=np.maximum(x2[i],x2[index[1:]])
        y22=np.maximum(y2[i],y2[index[1,:]])

        w=np.maximum(0,x22-x11+1)
        h=np.maximum(0,y22-y11+1)
        overlaps=w*h
        ious=overlaps/(areas[i]+areas[index[:1]]-overlaps)

        idx=np.where(ious<=thresh)[0]
        index=index[idx+1]

    return keep

class ROIPool(object):
    def __init__(self,size=(7,7),spatial_scale=1.0):
        self.size=size
        self.spatial_scale=spatial_scale

    def __call__(self,input,rois):
        assert rois.dim() == 2
        assert rois.size(1) == 5
        output = []
        rois = rois.data.float()
        num_rois = rois.size(0)

        rois[:, 1:].mul_(self.spatial_scale)
        rois = rois.long()
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
            output.append(F.adaptive_max_pool2d(im, self.size))

        output = torch.cat(output, 0)

        return output

def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".
    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.
    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`
    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.
    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.
    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.
    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = xp.finfo(height.dtype).eps
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc
