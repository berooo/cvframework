from yacs.config import CfgNode as CN
import argparse
import os
import sys

_C=CN()

_C.MODEL = CN()

_C.MODEL.NAME = 'resnet50'
_C.MODEL.NUM_CLASSES=124


_C.MODEL.POOL='gem'

_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "LinearHead"
# Normalization method for the convolution layers.
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 1000
# Input feature dimension
_C.MODEL.HEADS.IN_FEAT = 2048
# Reduction dimension in head
_C.MODEL.HEADS.REDUCTION_DIM = 512
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"
# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"
# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 30


_C.INPUT = CN()
_C.INPUT.SIZE_CROP=[448,448]
_C.INPUT.SIZE_INPUT=[480, 480]
_C.INPUT.SIZE_TEST=[448, 448]
_C.INPUT.PROB=0.5 # random horizontal flip
_C.INPUT.RE_PROB=0.5 # random erasing
_C.INPUT.PADDING=10
_C.INPUT.DROPOUTPORB= 0.5
_C.INPUT.DATAPATH='../../datasets/data/train'
_C.INPUT.CKPTPATH= '../../out/tricls/model_best.pyth'
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = True

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

_C.OPTIM = CN()

_C.OPTIM.BASE_LR= 0.001
_C.OPTIM.LR_POLICY= 'cos'
_C.OPTIM.STEPS=[0, 30, 60, 90]
_C.OPTIM.LR_MULT= 0.1
_C.OPTIM.MAX_EPOCH= 2000
_C.OPTIM.MOMENTUM= 0.9
_C.OPTIM.NESTEROV= True
_C.OPTIM.WEIGHT_DECAY= 0.0001
_C.OPTIM.WARMUP_EPOCHS= 5
# Momentum dampening
_C.OPTIM.DAMPENING = 0.0
# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 0.0001

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 5

_C.TRAIN = CN()
_C.TRAIN.DATASET=''
_C.TRAIN.SPLIT='train_list.txt'
_C.TRAIN.BATCH_SIZE=128
_C.TRAIN.IM_SIZE=512
_C.TRAIN.EVAL_PERIOD=100
_C.TRAIN.GPU=[0, 1, 2, 3, 4, 5, 6, 7, 8]

_C.TEST = CN()
_C.TEST.DATASET= 'GLDv2'
_C.TEST.SPLIT='val_list.txt'
_C.TEST.BATCH_SIZE=64
_C.TEST.IM_SIZE=256

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS= 4
_C.DATALOADER.NUM_INSTANCE= 6

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK=True

_C.OUT_DIR = './output_origin'

def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg",dest="cfg_file",default='retrieval/res50_tri_cls.yaml', help=help_s)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)
