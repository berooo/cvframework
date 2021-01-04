import os
from xml.etree.ElementInclude import default_loader

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
#from util.util import *
import torchvision.transforms as transforms
class ImagesFromList(Dataset):
  """A generic data loader that loads images from a list
      (Based on ImageFolder from pytorch)

  Args:
      root (string): Root directory path.
      images (list): Relative image paths as strings.
      imsize (int, Default: None): Defines the maximum size of longer image side
      bbxs (list): List of (x1,y1,x2,y2) tuples to crop the query images
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      images_fn (list): List of full image filename
  """

  def __init__(self, root, images, imsize=224, bbxs=None, transform=transforms.Compose([transforms.ToTensor()]), loader=default_loader):
    images_fn = [os.path.join(root, images[i]) for i in range(len(images))]

    if len(images_fn) == 0:
      raise (RuntimeError("Dataset contains 0 images!"))

    self.root = root
    self.images = images
    self.imsize = imsize
    self.images_fn = images_fn
    self.bbxs = bbxs
    self.transform = transform
    self.loader = loader

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        image (PIL): Loaded image
    """
    path = self.images_fn[index]
    img1 = Image.open(path).convert('RGB')
    if self.bbxs is not None:
      img1= img1.crop(self.bbxs[index])
    img = img1.resize((self.imsize, self.imsize))
    img = self.transform(img)

    return img

  def __len__(self):
    return len(self.images_fn)
