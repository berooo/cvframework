import traceback
import argparse
import query
from google.protobuf import json_format

from datasets.heatmapDataset import heatmapDataset
from datasets.imageListDateset import ImagesFromList
from datasets.siameseData import SiameseData
from option.evaluate import compute_map_and_print
from protos.train_pb2 import TrainConfig
from protos.model_pb2 import ModelConfig
from input import *
from graph import builGraph,buildLoss
from torch.autograd import Variable
import torchvision.transforms as transforms
from option.calMap import *
numpoints=25
sub_index=0


parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')
parser.add_argument('--trainconfig', metavar='trainConfig',default='config/retrieval/tripletparistrain.config',
                    help='destination where trained network should be saved')
parser.add_argument('--modelconfig', metavar='trainConfig',default='config/retrieval/tripletparismodel.config',
                    help='destination where trained network should be saved')

resultsList=[]

dataset_true = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

dataset_true['categories'].append({
    'id': 1,
    'name': "short_sleeved_shirt",
    'supercategory': "clothes",
    'keypoints': [str(i+1) for i in range(numpoints)],
    'skeleton': []
})


def tocpu(tensor):
    if isinstance(tensor,list):
        lis=[tocpu(i) for i in tensor]
    else:
        p=int(tensor.cpu().numpy()[0])
        return p
    return lis

def writeTotruejsons(d,true_pos,landmark_vis):
    global sub_index
    image_name = '/data00/home/shibaorong/testCOCO/train/image/' + d['image_id'][0] + '.jpg'
    imag = Image.open(image_name)
    width, height = imag.size
    dataset_true['images'].append({
        'coco_url': '',
        'date_captured': '',
        'file_name': d['image_id'][0] + '.jpg',
        'flickr_url': '',
        'id': d['image_id'][0],
        'license': 0,
        'width': width,
        'height': height
    })

    points = np.zeros(numpoints * 3)
    sub_index = sub_index + 1
    box = tocpu(d['bounding_box'])
    w = box[2] - box[0]
    h = box[3] - box[1]
    x_1 = box[0]
    y_1 = box[1]
    bbox = [x_1, y_1, w, h]
    cat = tocpu(d['category_id'].int())
    style = tocpu(d['style'].int())
    seg = tocpu(d['segmentation'])
    landmarks = true_pos

    points_p = landmarks.cpu().numpy()
    points_v=landmark_vis


    for n in range(0, numpoints):
        points[3 * n] = points_p[0][n][0]
        points[3 * n + 1] = points_p[0][n][1]
        points[3 * n + 2] = points_v[0][n][0]
    num_points = len(np.where(points_v > 0)[0])
    dataset_true['annotations'].append({
        'area': w * h,
        'bbox': bbox,
        'category_id': cat,
        'id': sub_index,
        'pair_id': tocpu(d['pair_id'].int()),
        'image_id': d['image_id'][0],
        'iscrowd': 0,
        'style': style,
        'num_keypoints': num_points,
        'keypoints': points.tolist(),
        'segmentation':seg,
    })


def writeToResultJsons(d,predict_pos,landmark_vis):
    record={
    'image_id':d['image_id'][0],
    'category_id':tocpu(d['category_id'].int()),
    'keypoints':[],
    'score':1.0
    }

    landmarks = predict_pos

    points = np.zeros(numpoints * 3)
    points_p = landmarks
    points_v = landmark_vis

    for n in range(0, numpoints):
        points[3 * n] = points_p[0][n][0]
        points[3 * n + 1] = points_p[0][n][1]
        points[3 * n + 2] = points_v[0][n][0]

    record['keypoints']=points.tolist()
    return record

class Evaluator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.lm_vis_count_all=np.array([0.]*numpoints)
        self.lm_dist_all=np.array([0.]*numpoints)

    def add(self,output,sample):
        landmark_vis_count=sample['landmark_vis'].cpu().numpy().sum(axis=0)
        landmark_vis_float=torch.unsqueeze(sample['landmark_vis'].float(),dim=2)
        landmark_vis_float=torch.cat([landmark_vis_float,landmark_vis_float],dim=2).cpu().detach().numpy()

        lm_pos_map=output['lm_pos_map']
        batchsize,_,pred_h,pred_w=lm_pos_map.size()
        lm_pos_reshaped=lm_pos_map.reshape(batchsize,numpoints,-1)
        lm_pos_y,lm_pos_x=np.unravel_index(torch.argmax(lm_pos_reshaped,dim=2).cpu().numpy(),(pred_h,pred_w))
        lm_pos=np.stack([lm_pos_x,lm_pos_y],axis=2)[0]
        image=sample['image'][0]

        #lm_pos_output=np.stack([lm_pos_x/(pred_w-1),lm_pos_y/(pred_h-1)],axis=2)
        lm_pos_output = np.stack([lm_pos_x, lm_pos_y], axis=2)
        self.predict_pos=lm_pos_output
        self.true_pos=sample['landmark_pos'].int()
        self.landmark_vis=landmark_vis_float
        lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
        landmark_dist=np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float*lm_pos_output-landmark_vis_float*sample['landmark_pos'].cpu().numpy()
        ),axis=2)),axis=0)

        self.lm_vis_count_all+=landmark_vis_count
        self.lm_dist_all+=landmark_dist

    def evaluate(self):
        lm_dist=self.lm_dist_all/self.lm_vis_count_all
        lm_dist[np.isnan(lm_dist)] = 0
        lm_dist_all=lm_dist.mean()

        return {'lm_dist':lm_dist,
                'lm_dist_all':lm_dist_all}

    def getpredictpos(self):
        return self.predict_pos,self.true_pos,self.landmark_vis

def testClassification(params,transform):
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

        for batch_x,batch_y in mytrainloader:
            train_loss = 0.
            train_acc = 0.
            if cuda_gpu:
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()
            batch_x=batch_x.float()
            batch_x,batch_y=Variable(batch_x),Variable(batch_y)

            out=mymodel(batch_x)
            print(out.data.size())

            loss=builGraph.getloss(out,batch_y)

            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prediction=torch.argmax(out,1)
            train_acc+=(prediction==batch_y).sum().float()
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(batch_x)), train_acc / (len(batch_x))))
            torch.save({'epoch': epoch,
                        'model_state_dict': mymodel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, params['train_dir'])

def testOnlinepair(params,transform):
    mytraindata = myDataset(path=params['data_dir'], height=params['height'], width=params['width'],
                            autoaugment=params['autoaugment'], transform=transform)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)
    gnd = loadquery(params['valdata_dir'])
    cuda_gpu = torch.cuda.is_available()
    '''mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                                 params['model_type'],cuda_gpu=cuda_gpu)'''
    mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                                 'triplet', cuda_gpu=cuda_gpu)

    if os.path.exists(params['train_dir']):
        checkpoint=torch.load(params['train_dir'])
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    mymodel.eval()
    with torch.no_grad():
        print('>> Extracting descriptors for query images...')
        qloader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[i['queryimgid'] for i in gnd], imsize=mytraindata.height,
                           transform=mytraindata.transform),
            batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )
        qoolvecs = torch.zeros(params['class_num'], len(gnd)).cuda()
        lenq=len(qloader)
        for i, input in enumerate(qloader):
            out, _,_ = mymodel(input.cuda(),input.cuda(),input.cuda())
            qoolvecs[:, i] = out[0].data.squeeze()
            if (i + 1) % 10 == 0:
                print('\r>>>> {}/{} done...'.format(i + 1, lenq), end='')
        print('')

        poolvecs = torch.zeros(params['class_num'], len(mytrainloader)).cuda()
        idlist=[]
        for index, data in enumerate(mytrainloader):
            batch_x, batch_y, batch_id = data
            idlist.append(batch_id[0])
            if cuda_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            batch_x = batch_x.float()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out,_,_ = mymodel(batch_x,batch_x,batch_x)
            poolvecs[:,index]=out[0]

        vecs = poolvecs.cpu().numpy()
        qvecs = qoolvecs.cpu().numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        dataset = params['data_dir'].split('/')[-1].replace("train.json", "")
        compute_map_and_print(dataset, ranks, gnd,idlist)

    '''relu_ip1_list = []
    logits_list = []
    id_list = []
    label_list = []
    sstart = time.clock()
    mymodel.eval()
    with torch.no_grad():
        for index, data in enumerate(mytrainloader):
            batch_x, batch_y, batch_id = data
            if cuda_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            batch_x = batch_x.float()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out, features = mymodel(batch_x)
            reluip = features['ip1'].cpu().numpy()
            logits = out.cpu().numpy()

            binarvalues = toBinaryString(logits)

            for binary in binarvalues:
                logits_list.append(binary)

            for unit in reluip:
                relu_ip1_list.append(unit)

            for id in batch_id:
                id_list.append(id)

            for label in batch_y:
                label_list.append(label.cpu())

    featurelib=query.getFeatureLib()
    evalex = eval(id_list, logits_list, relu_ip1_list, label_list, featurelib)
    evalex.extract()
    top1, top5, MAP = evalex.calMAP()
    eend = time.clock()
    print(len(id_list))
    print('top1:%f' % top1)
    print('top5:%f' % top5)
    print('MAP:%f' % MAP)
    print(str(eend - sstart))'''


def testSiamese(params,transform):
    mytraindata = SiameseData(path=params['data_dir'], height=params['height'], width=params['width'],
                            autoaugment=params['autoaugment'], transform=transform)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=params['BATCH_SIZE'], shuffle=True)

    cuda_gpu = torch.cuda.is_available()
    mymodel = builGraph.getModel(params['modelName'], params['featuredim'], params['Gpu'],
                                 params['model_type'], cuda_gpu=cuda_gpu)

    if params['train_method'] == 'gd':
        optimizer = torch.optim.SGD(mymodel.parameters(), lr=params['LR'])
    else:
        optimizer = torch.optim.Adam(mymodel.parameters())

    startepoch = 0
    if os.path.exists(params['train_dir']):
        checkpoint = torch.load(params['train_dir'])
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startepoch = checkpoint['epoch']

    for epoch in range(startepoch, params['maxepoch']):
        print('epoch {}'.format(epoch + 1))

        for image1,image2,label in mytrainloader:
            train_loss = 0.
            train_acc = 0.
            if cuda_gpu:
                image1 = image1.cuda()
                image2 = image2.cuda()
                label = label.cuda()
            image1 = image1.float()
            image2=image2.float()
            image1,image2, label = Variable(image1), Variable(image2),Variable(label)

            out1,out2 = mymodel(image1,image2)

            out=[out1,out2]
            loss = buildLoss.getSiameseloss(out1,out2,label)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #prediction = torch.argmax(out, 1)
            #train_acc += (prediction == batch_y).sum().float()
            print('Train Loss: {:.6f}'.format(train_loss / (len(image1))))
            torch.save({'epoch': epoch,
                        'model_state_dict': mymodel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, params['train_dir'])


def testTriplet(params,transform):
    mytraindata = OnlineTripletData(path=params['data_dir'], autoaugment=params['autoaugment'],
                                    outputdim=params['class_num'],
                                    imsize=params['height'], transform=transform)
    cuda_gpu = torch.cuda.is_available()
    miningmodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                                     'triplet', cuda_gpu=cuda_gpu)
    gnd=loadquery(params['valdata_dir'])

    if os.path.exists(params['train_dir']):
        checkpoint = torch.load(params['train_dir'])
        miningmodel.load_state_dict(checkpoint['model_state_dict'])

    miningmodel.eval()

    with torch.no_grad():
        print('>> Extracting descriptors for query images...')
        qloader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[i['queryimgid'] for i in gnd], imsize=mytraindata.imsize,
                           transform=mytraindata.transform),
            batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )
        qoolvecs = torch.zeros(params['class_num'], len(gnd)).cuda()
        for i, input in enumerate(qloader):
            out, _ = miningmodel(input.cuda())
            qoolvecs[:, i] = out.data.squeeze()
            if (i + 1) % mytraindata.print_freq == 0 or (i + 1) == mytraindata.qsize:
                print('\r>>>> {}/{} done...'.format(i + 1, mytraindata.qsize), end='')
        print('')

        print('>> Extracting descriptors for data images...')
        dloader= torch.utils.data.DataLoader(
        ImagesFromList(root='', images=[i['filenames'] for i in mytraindata.data], imsize=mytraindata.imsize,
                       transform=mytraindata.transform),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
      )
        poolvecs = torch.zeros(params['class_num'], len(mytraindata.data)).cuda()
        idlist=[i['filenames'] for i in mytraindata.data]
        for i, input in enumerate(dloader):
            out, _ = miningmodel(input.cuda())
            poolvecs[:, i] = out.data.squeeze()
            if (i + 1) % mytraindata.print_freq == 0 or (i + 1) == mytraindata.qsize:
                print('\r>>>> {}/{} done...'.format(i + 1, mytraindata.qsize), end='')
        print('')

        vecs = poolvecs.cpu().numpy()
        qvecs = qoolvecs.cpu().numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        dataset=params['data_dir'].split('/')[-1].replace("train.json","")
        compute_map_and_print(dataset, ranks, gnd,idlist)

def testLandmark(params,transform):
    mytraindata = heatmapDataset(path=params['data_dir'], height=params['height'], width=params['width'],
                              autoaugment=params['autoaugment'], transform=transform)
    mytrainloader = torch.utils.data.DataLoader(mytraindata, batch_size=1, shuffle=False)

    cuda_gpu = torch.cuda.is_available()
    mymodel = builGraph.getModel(params['modelName'], params['class_num'], params['Gpu'],
                                 params['model_type'], cuda_gpu=cuda_gpu)


    if os.path.exists(params['train_dir']):
        checkpoint = torch.load(params['train_dir'])
        mymodel.load_state_dict(checkpoint['model_state_dict'])

    mymodel.eval()
    test_step=len(mytrainloader)
    print('Evaluating.....')
    with torch.no_grad():
        evaluator=Evaluator()
        for i,sample in enumerate(mytrainloader):

            for key in sample:
                if isinstance(sample[key],list):
                    continue
                sample[key]=sample[key].cuda().float()
            out=mymodel(sample)

            evaluator.add(out,sample)
            predict_pos,true_pos,landmark_vis=evaluator.getpredictpos()
            writeTotruejsons(sample,true_pos,landmark_vis)
            resultsList.append(writeToResultJsons(sample,predict_pos,landmark_vis))
            print('Val Step [{}/{}]'.format(i + 1, test_step))

            if i>500:
                break
        #results=evaluator.evaluate()

        json_name = 'short_sleeve_top_true.json'
        with open(json_name, 'w') as f:
            json.dump(dataset_true, f)
        json_name = 'short_sleeve_top_test.json'
        with open(json_name, 'w') as f:
            json.dump(resultsList, f)

        '''print(
            '|  L.Collar  |  R.Collar  |  L.Sleeve  |  R.Sleeve  |   L.Waist  |   R.Waist  |    L.Hem   |   R.Hem    |     ALL    |')
        print(
            '|   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |{:.5f}  |   {:.5f}  |'
            .format(results['lm_dist'][0], results['lm_dist'][1], results['lm_dist'][2], results['lm_dist'][3],
                    results['lm_dist'][4], results['lm_dist'][5], results['lm_dist'][6], results['lm_dist'][7],
                    results['lm_dist'][8], results['lm_dist'][9], results['lm_dist_all']))'''


def test(train_config,model_config):

    params={}

    params['train_dir']=train_config.train_dir
    params['data_dir']=train_config.data_dir
    params['valdata_dir'] = train_config.valdata_dir
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
        testClassification(params,transform)
    elif params['model_type']=='siamese':
        testSiamese(params,transform)
    elif params['model_type']=='triplet':
        testTriplet(params,transform)
    elif params['model_type']=='onlinepair':
        testOnlinepair(params,transform)
    elif params['model_type']=='landmark':
        testLandmark(params,transform)
    else:
        raise Exception("modeltype doesn't exist!")


def main():
    global args
    args = parser.parse_args()
    trainConfig = args.trainconfig
    modelConfig = args.modelconfig

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
    except Exception as e:
        print(e)
        print('error when parsing %s and %s.',trainConfig,modelConfig)

    test(train_config,model_config)

if __name__=='__main__':
    main()