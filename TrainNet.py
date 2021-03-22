import traceback

import argparse
import time
from google.protobuf import json_format
from protos.train_pb2 import TrainConfig
from protos.model_pb2 import ModelConfig
from input import *
from graph import builGraph,buildLoss
from torch.autograd import Variable
import torchvision.transforms as transforms
from util.util import to_Onehot
from network.outputdim import OUTPUT_DIM

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')
parser.add_argument('--trainconfig', metavar='trainConfig',default='config/retrieval/tripletdshtrain.config',
                    help='tripletdshtrain,traindshNet')
parser.add_argument('--modelconfig', metavar='trainConfig',default='config/retrieval/tripletdshmodel.config',
                    help='destination where trained network should be saved')



def trainClassification(params,transform):
    mytraindata = myDataset(path=params['data_dir'], height=params['height'], width=params['width'], autoaugment=params['autoaugment'], transform=transform)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=params['BATCH_SIZE'], shuffle=True)

    cuda_gpu=torch.cuda.is_available()
    mymodel = builGraph.getModel(params['modelName'],params['class_num'], params['Gpu'],
                                 params['model_type'],cuda_gpu=cuda_gpu)

    if params['train_method']=='gd':
        optimizer=torch.optim.SGD(mymodel.parameters(),lr=params['LR'])
    else:
        optimizer=torch.optim.Adam(mymodel.parameters())

    startepoch = 0
    if os.path.exists(params['train_dir']):
        checkpoint = torch.load(params['train_dir'])
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']

    for epoch in range(startepoch, params['maxepoch']):
        print('epoch {}'.format(epoch + 1))

        for batch_x,batch_y,_ in mytrainloader:
            train_loss = 0.
            train_acc = 0.
            if cuda_gpu:
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()
            batch_x=batch_x.float()
            batch_x,batch_y=Variable(batch_x),Variable(batch_y)

            out,_=mymodel(batch_x)


            loss=buildLoss.getloss(out,batch_y)

            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            prediction=torch.argmax(out,1)
            train_acc+=(prediction==batch_y).sum().float()
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(batch_x)), train_acc / (len(batch_x))))


            torch.save({'epoch': epoch,
                        'model_state_dict': mymodel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, params['train_dir'])

def trainOnlinepair(params,transform):
    minloss=float("inf")
    mytraindata = myDataset(path=params['data_dir'], height=params['height'], width=params['width'],
                            autoaugment=params['autoaugment'], transform=transform)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=params['BATCH_SIZE'], shuffle=True)

    cuda_gpu = torch.cuda.is_available()
    mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                                 params['model_type'],cuda_gpu=cuda_gpu)

    if params['train_method'] == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=params['LR'])
    else:
        optimizer = torch.optim.Adam(mymodel.parameters(),lr=0.001)

    startepoch=0
    if os.path.exists(params['train_dir']):
        checkpoint=torch.load(params['train_dir'])
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']
        minloss=checkpoint['loss']

    lendata=len(mytrainloader)

    for epoch in range(startepoch,params['maxepoch']):
        print('epoch {}'.format(epoch + 1))

        train_loss = 0.
        for index,data in enumerate(mytrainloader):

            batch_x, batch_y, _=data
            batch_y=to_Onehot(batch_y,params['class_num'])

            if cuda_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            batch_x = batch_x.float()
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out,_ = mymodel(batch_x)

            loss = buildLoss.getDshloss(out, batch_y,params['class_num'])

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (index+1)%10==0:
                print('Train Loss: {:.6f}'.format(loss))

        pklword = params['train_dir'].split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (epoch + 1)
        path = params['train_dir'].replace(pklword, newpkl)
        train_loss=train_loss/lendata
        is_best=train_loss<minloss
        save_checkpoint({'epoch': epoch,
                        'model_state_dict': mymodel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss
                        },is_best,path)
        print('epoch Train Loss: {:.6f}'.format(train_loss))




def trainTriplet(params,transform):
    mytraindata = OnlineTripletData(path=params['data_dir'], autoaugment=params['autoaugment'],outputdim=params['class_num'],
                                    imsize=params['height'], transform=transform)
    #mytraindata = myDataset(path=params['data_dir'], autoaugment=params['autoaugment'],transform=transform)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=params['BATCH_SIZE'], shuffle=True)

    cuda_gpu = torch.cuda.is_available()
    minloss=float("inf")
    #mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
    #                             params['model_type'], cuda_gpu=cuda_gpu)

    miningmodel=builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                                 'onlinepair', cuda_gpu=cuda_gpu)
    if params['train_method'] == 'gd':
        optimizer = torch.optim.SGD(miningmodel.parameters(), lr=params['LR'])
    else:
        optimizer = torch.optim.Adam(miningmodel.parameters())

    startepoch = 0
    if os.path.exists(params['train_dir']):
        checkpoint = torch.load(params['train_dir'])
        miningmodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']
        minloss = checkpoint['loss']



    for epoch in range(startepoch, params['maxepoch']):

        print('epoch {}'.format(epoch + 1))


        record = 0
        mytrainloader.dataset.create_epoch_tuples(miningmodel)
        print(minloss)
        miningmodel.train()
        miningmodel.apply(set_batchnorm_eval)
        tloss=0.
        for i, (input, plabel, nlabel) in enumerate(mytrainloader):
            if input is None or plabel is None or nlabel is None:
                continue
            train_loss = 0.
            iter_start_time=time.time()
            nq = len(input[0])
            ni = len(input)
            optimizer.zero_grad()
            for q in range(nq):

                output = torch.zeros(params['class_num'], ni).cuda()
                f = torch.zeros(OUTPUT_DIM[params['modelName']], ni).cuda()
                for imi in range(ni):
                    # compute output vector for image imi
                    in_ = Variable(input[imi][q].unsqueeze(0).cuda().float())
                    out, features = miningmodel(in_)
                    f[:, imi] = features['ip1'].squeeze()
                    output[:, imi] = out.squeeze()
                p = plabel[q]
                n = nlabel[q]

                loss = buildLoss.getTripletloss(output, f, p, n)
                train_loss += loss.item()

                loss.backward()
                if (q + 1) % 10 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            torch.cuda.empty_cache()
            t = time.time() - iter_start_time
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}], Loss: {:.4f}, Time:{:.3f}'.format(epoch + 1, params['maxepoch'], i + 1,
                                                                                   train_loss, t))
            tloss += train_loss
            record += 1
        tloss/=record
        pklword = params['train_dir'].split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (epoch + 1)
        path = params['train_dir'].replace(pklword, newpkl)
        is_best = tloss < minloss
        if is_best:
            minloss=tloss
        save_checkpoint({'epoch': epoch,
                         'model_state_dict': miningmodel.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': tloss
                         }, is_best, path)
        print('epoch Train Loss: {:.6f}'.format(tloss))
        '''record = index
        train_acc = 0.

        if cuda_gpu:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        batch_x = batch_x.float()
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        optimizer.zero_grad()
        out,_ = miningmodel(batch_x)

        loss = buildLoss.getloss(out, batch_y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        prediction = torch.argmax(out, 1)
        train_acc += (prediction == batch_y).sum().float()
        acc = train_acc / len(batch_x)
        trainacc += acc
        if (index + 1) % 10 == 0:
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(loss, acc))
    train_loss = train_loss / lendata
    train_acc = trainacc / (record + 1)

        print('epoch Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss, train_acc))
        pklword = params['train_dir'].split('/')[-1]
        newpkl = 'parameter_%02d.pkl' % (epoch + 1)
        path = params['train_dir'].replace(pklword, newpkl)

        is_best = train_loss < minloss
        save_checkpoint({'epoch': epoch,
                         'model_state_dict': miningmodel.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': train_loss
                         }, is_best, path)'''






def trainLandmark(params,transform):
    mytraindata = heatmapDataset(path=params['data_dir'], height=params['height'], width=params['width'],
                              autoaugment=params['autoaugment'], transform=transform)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=params['BATCH_SIZE'], shuffle=True)

    cuda_gpu = torch.cuda.is_available()
    mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                                 params['model_type'], cuda_gpu=cuda_gpu)
    optimizer = torch.optim.Adam(mymodel.parameters())

    startepoch = 0
    if os.path.exists(params['train_dir']):
        checkpoint = torch.load(params['train_dir'])
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']

    lendata=len(mytrainloader)
    for epoch in range(startepoch, params['maxepoch']):
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.
        for i,sample in enumerate(mytrainloader):
            iter_start_time=time.time()
            for key in sample:
                if isinstance(sample[key],list):
                    continue
                sample[key]=sample[key].cuda().float()
            out=mymodel(sample)
            loss=buildLoss.getGLEMloss(sample,out,params['class_num'])
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t = time.time() - iter_start_time
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}], Loss: {:.4f}, Time:{:.3f}'.format(epoch + 1, params['maxepoch'], i + 1, loss.item(), t))
                pklword = params['train_dir'].split('/')[-1]
                newpkl = 'parameter_%02d.pkl' % (epoch + 1)
                path = params['train_dir'].replace(pklword, newpkl)
                torch.save({'epoch': epoch,
                        'model_state_dict': mymodel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, path)
        train_loss=train_loss/(lendata)
        print('epoch Train Loss: {:.6f}'.format(train_loss))

def traindelf(params,transform):
    os.environ['CUDA_VISIBLE_DEVICES']=params['Gpu']
    use_cuda=torch.cuda.is_available()
    stage=params['stage']

    if stage in ['keypoint']:
        pass
    elif stage in ['finetune']:

        pass

    train_loader_pt,val_loader_pt=get_loader(
        train_path=params['train_dir'],
        val_path=params['val_dir'],
        stage=stage,
        train_batch_size=params['BATCH_SIZE'],
        val_batch_size=params['val_batch_size'],
        sample_size=params['sample_size'],
        crop_size=params['crop_size'],
        workers=0
    )

    train_loader_ft, val_loader_ft = get_loader(
        train_path=params['train_dir_finetuning'],
        val_path=params['val_dir_finetuning'],
        stage=stage,
        train_batch_size=params['BATCH_SIZE'],
        val_batch_size=params['val_batch_size'],
        sample_size=params['sample_size'],
        crop_size=params['crop_size'],
        workers=0
    )

    #load model
    from network.retrieval import delf


def train(train_config,model_config):

    params={}

    params['train_dir']=train_config.train_dir
    params['data_dir']=train_config.data_dir
    params['LR']=train_config.initial_learning_rate
    params['train_method']=train_config.optimizer
    params['class_num']=model_config.num_classes

    params['modelName']=model_config.backbone
    params['height']=model_config.height
    params['width']=model_config.width
    params['BATCH_SIZE']=train_config.batch_size
    params['maxepoch']=train_config.max_epochs
    params['autoaugment']=model_config.autoaugment
    params['Gpu']=train_config.gpus
    params['model_type']=model_config.modeltype

    transform = transforms.Compose(
        [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if params['model_type']=='base':
        trainClassification(params,transform)
    elif params['model_type']=='siamese':
        trainSiamese(params,transform)
    elif params['model_type']=='triplet':
        trainTriplet(params,transform)
    elif params['model_type']=='onlinepair':
        trainOnlinepair(params,transform)
    elif params['model_type']=='landmark':
        trainLandmark(params,transform)
    else:
        raise Exception("modeltype doesn't exist!")


def main():
    global args
    args=parser.parse_args()
    trainConfig=args.trainconfig
    modelConfig=args.modelconfig

    assert trainConfig != ''
    assert modelConfig != ''

    try:
        train_config = TrainConfig()
        model_config = ModelConfig()

        with open(trainConfig, 'r') as f:
            info = f.read()
            json_format.Parse(info, train_config)

        with open(modelConfig, 'r') as f:
            info = f.read()
            json_format.Parse(info, model_config)
    except:
        traceback.print_ex()
        print('error when parsing %s and %s.',trainConfig,modelConfig)

    train(train_config,model_config)


if __name__=='__main__':
    main()