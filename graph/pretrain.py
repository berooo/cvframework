#coding:utf-8

from torchvision import transforms

def type1(crop_size=256,sample_size=224):
  transformList = []
  normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.225, 0.225, 0.225]
  )
  transformList.append(transforms.CenterCrop(crop_size))
  transformList.append(transforms.RandomCrop(sample_size))
  transformList.append(transforms.RandomHorizontalFlip())
  transformList.append(transforms.ToTensor())
  transformList.append(normalize)
  transformSequence = transforms.Compose(transformList)

  return transformSequence


