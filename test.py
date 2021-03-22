import torch
from torch.nn import Parameter
from torch import nn
from torch.utils.data import Dataset,DataLoader
import numpy as np

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.W=Parameter(torch.Tensor([1]))
        self.b=Parameter(torch.Tensor([1]))

    def forward(self,x):
        return self.W*x+self.b

class LRData(Dataset):
    def __init__(self,x):

        self.x=x

        self.y=[3*i+2 for i in x]

    def __getitem__(self, item):
        xs=np.asarray(self.x[item])
        ys=np.asarray(self.y[item])
        xs=torch.from_numpy(xs)
        ys=torch.from_numpy(ys)
        return xs,ys

    def __len__(self):
        return len(self.x)

def setup_model():
    my=model()
    return my

def train(my,dataloader,maxepoch,loss_func,optimizer):

    for i in range(maxepoch):
        for index,(x,y) in enumerate(dataloader):
            x=x.float()
            ry=my(x)
            y=y.float()
            optimizer.zero_grad()
            loss=loss_func(ry,y)
            print(loss)
            loss.backward()
            optimizer.step()


if __name__=='__main__':
    my=setup_model()
    maxepoch=100
    x=[i for i in range(1000)]
    loss_func=torch.nn.MSELoss()
    dataset=LRData(x)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    optimizer=torch.optim.Adam(my.parameters())
    train(my,dataloader,maxepoch,loss_func,optimizer)
    print(my.W)
    print(my.b)