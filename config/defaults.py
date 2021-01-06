from yacs.config import CfgNode as CN
import argparse
import os
import sys

_C=CN()

_C.MODEL = CN()
_C.MODEL.LOSSES = CN()
_C.MODEL.HEADS = CN()

_C.INPUT = CN()

_C.OPTIM = CN()
_C.TRAIN = CN()
_C.TEST = CN()
_C.DATA_LOADER = CN()
_C.CUDNN = CN()


_C.OUT_DIR = CN()

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
