import sys



sys.path.insert(0, '../')
import os
import torch
import numpy as np
from PIL import Image
from graph import builGraph
from scipy.io import savemat
from extract.multinet_utils import MultinetExtraction
import torchvision.transforms as transforms
#读query
#读gallery
#获取model
#抽取特征
#特征rank
#获取映射id
#写入csv

imgdir='/home/shibaorong/cartoon/extraction/data/cartoontest'
checkpoints='/mnt/sdb/shibaorong/logs/cartoon/checkpoints/model_best.pyth'
gallery_path='../data/cartoon/gallery.txt'
query_path='../data/cartoon/query.txt'
tofile='features'

NUM_EMBEDDING_DIMENSIONS=512
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def setup_model():
    model=MultinetExtraction('vgg16')
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    load_checkpoint(checkpoints,model)

    model.eval()
    return model

def load_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def get_img_name(path,imgdir):
    imgpaths=[]
    for line in open(path):
        imgpaths.append(os.path.join(imgdir,line.strip()+'.jpg'))
    return imgpaths


def preprocess(img):
    img=Image.open(img).convert('RGB')
    img=transform(img)
    img=img.unsqueeze(0)
    return img

def generate_embedding(model,imgpaths,mode):

    num_embeddings=len(imgpaths)
    embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))
    for i, image_path in enumerate(imgpaths):

        input_data = preprocess(image_path)
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        feature=model(input_data,mode)

        embeddings[i, :] = feature.cpu().detach().numpy()
        print(str(i) + ',' + str(len(imgpaths)))

    return embeddings


def main():
    querys=get_img_name(query_path,imgdir)
    gallerys=get_img_name(gallery_path,imgdir)
    model=setup_model()
    '''model=builGraph.getModel('vgg16', 124, [0,1],
                                 'retrieval', cuda_gpu=True,pretrained=True)'''
    query_embeddings=generate_embedding(model,querys,mode='c')
    gallery_embeddings=generate_embedding(model,gallerys,mode='p')
    savemat(os.path.join(tofile,'cartoon.mat'),{'C':query_embeddings,'P':gallery_embeddings})

if __name__=='__main__':
    main()



