import os

import torch
from torch.autograd import Variable

from datasets.siameseData import SiameseData
from graph import builGraph, buildLoss


def trainSiamese(params, transform):
  mytraindata = SiameseData(path=params['data_dir'], height=params['height'], width=params['width'],
                            autoaugment=params['autoaugment'], transform=transform)
  mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=params['BATCH_SIZE'], shuffle=True)

  cuda_gpu = torch.cuda.is_available()
  mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                               params['model_type'], cuda_gpu=cuda_gpu)

  if params['train_method'] == 'gd':
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=params['LR'])
  else:
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)

  startepoch = 0
  if os.path.exists(params['train_dir']):
    checkpoint = torch.load(params['train_dir'])
    mymodel.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    startepoch = checkpoint['epoch']

  for epoch in range(startepoch, params['maxepoch']):
    print('epoch {}'.format(epoch + 1))

    for image1, image2, label in mytrainloader:
      train_loss = 0.
      train_acc = 0.
      if cuda_gpu:
        image1 = image1.cuda()
        image2 = image2.cuda()
        label = label.cuda()
      image1 = image1.float()
      image2 = image2.float()
      image1, image2, label = Variable(image1), Variable(image2), Variable(label)

      out1, out2 = mymodel(image1, image2)

      out = [out1, out2]
      loss = buildLoss.getSiameseloss(out1, out2, label)

      train_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # prediction = torch.argmax(out, 1)
      # train_acc += (prediction == batch_y).sum().float()
      print('Train Loss: {:.6f}'.format(train_loss / (len(image1))))
      torch.save({'epoch': epoch,
                  'model_state_dict': mymodel.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss
                  }, params['train_dir'])