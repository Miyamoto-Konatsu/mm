# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from pathlib import Path
import scipy.io
import yaml
import math
import copy
import ours_datasets
from model_t import *
from utils import fuse_all_conv_bn

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./target/image',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

# 读取配置文件
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 500

if 'linear_num' in config:
    opt.linear_num = config['linear_num']

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# 读数据

h, w = 224, 224
data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir


image_datasets = {x: ours_datasets.OursDataset( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=0) for x in ['gallery','query']}

use_gpu = torch.cuda.is_available()


# 读取模型

def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


# 提取gallery和query的feature

def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, dataloaders):
    count = 0

    for iter, data in enumerate(dataloaders):
        img, label, texts = data
        classes_list = dataloaders.dataset.classes
        class_index = label

        true_label = []
        for now_class in class_index:
            true_label.append(classes_list[now_class])
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            texts = Variable(texts.cuda())
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img, texts)
                # text_dir = test_dir.replace('images', 'image_text')
                # text_dir = text_dir + '/' + type
                #
                # current_label = true_label[0]
                # current_text_dir = text_dir + '/' + current_label
                # first_file = current_text_dir + '/' + os.listdir(current_text_dir)[0]
                # with open(first_file) as vec_file:
                #     vec = np.array(vec_file.read().splitlines(), dtype=np.float32).reshape((100, 1)).T
                #     ALL_VEC = copy.deepcopy(vec)
                #
                #
                #
                # for current_label in true_label[1:]:
                #     current_text_dir = text_dir + '/' + current_label
                #     first_file = current_text_dir + '/' + os.listdir(current_text_dir)[0]
                #     with open(first_file) as vec_file:
                #         # vec = float(vec_file.read())
                #         vec = np.array(vec_file.read().splitlines(), dtype=np.float32).reshape((100, 1)).T
                #         ALL_VEC = np.vstack([ALL_VEC, vec])
                # ALL_VEC = torch.from_numpy(ALL_VEC)
                # ALL_VEC = ALL_VEC.cuda()
                # outputs = torch.cat((outputs, ALL_VEC), dim=1)
                ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        start = iter * opt.batchsize
        end = min( (iter+1) * opt.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
    return features

def get_id(img_path):
    labels = []
    for path_ in img_path:
        label = str(Path(path_).parent)[-6:]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
    return labels

#根据图片的名字提取出商品实际的label，就是商品的ID
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_label = get_id(gallery_path)
query_label = get_id(query_path)

model_structure = Model(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
model = load_network(model_structure)

# 去掉最后的分类层，只提取特征

model.classifier.classifier = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()

model = fuse_all_conv_bn(model)

since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'] )
    query_feature = extract_feature(model,dataloaders['query'])
time_elapsed = time.time() - since
print('Complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
# 保存我们提取出的特征
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'query_f':query_feature.numpy(),'query_label':query_label}
scipy.io.savemat('pytorch_result.mat',result)

