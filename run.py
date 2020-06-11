#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:24:56 2020

@author: cat
"""

import torch, pickle, os, torchvision
from models import YOLOr
import numpy as np
from network import DEF_yolo
import dataset as D
from sklearn.model_selection import KFold
"""
DEF_yolo={  'FilterSize':3,
            'FiltersNumHighRes':np.array([8, 16, 32]),
            'FiltersNumLowRes':np.array([16, 32, 64]),
            'FiltersDecoder':np.array([16, 32, 64]),
            'Categories':int(3), 
            'Activation':nn.LeakyReLU, 
            'InblockSkip':False,
            'ResidualConnections':False,
            'PoolShape':2,
            'BNorm':nn.BatchNorm3d,
            'Conv':nn.Conv3d,
            'Downsample':PoolWrapper,
            'Upsample':TransposeWrapper,
            'InterpMode':'trilinear',
            'DownConvKernel':3,
            'TransposeSize':4,
            'TransposeStride':2,
            'PositiveWeight':10,
            'CoordsWeight':1}
"""

dataroot='/home/cat/ADAM_release_subjs'
saveroot='/media/Olowoo/ADAMsaves/'
name='withcoords'
groupfile='groups.p'

saveprogress=saveroot+name+'progress.pth'
savebest=saveroot+name+'bestmodel.pth'
torch.set_default_tensor_type('torch.cuda.FloatTensor') # t
torch.backends.cudnn.benchmark = True
fold=0
Bsize=1

if os.path.isfile(groupfile):
    groups=pickle.load(open(groupfile,'rb'))
else:
    
    datasamples=[k for k in D.listdata(dataroot).keys()]
    shuffled=np.random.choice(datasamples,len(datasamples),replace=False)
    
    groups={0:[],1:[],2:[],3:[],4:[]}
    idx=0
    for k in shuffled:
        groups[idx].append(k)
        idx+=1
        if idx==5: idx=0
    pickle.dump(groups, open(groupfile,'wb'))

shift=D.Shift([40,15,0],probability=0.5,order={'sample':3,'labels':0})
rot=D.Rotate(5, probability=0.5,order={'sample':3,'labels':0})
norm=D.Normalize(order={'sample':3,'labels':0})
tensor=D.ToTensor(order={'sample':3,'labels':0,'coords':None})


train_transforms = torchvision.transforms.Compose([rot,shift,norm,tensor])
test_transforms = torchvision.transforms.Compose([norm,tensor])

train_dataset=D.YoloDataset(dataroot,train_transforms,IDs=D.IDListMaker(groups,fold,'train'))
test_dataset =D.YoloDataset(dataroot,test_transforms,IDs=D.IDListMaker(groups, fold,'test'))

trainloader=torch.utils.data.DataLoader(train_dataset, batch_size=Bsize,shuffle=True,num_workers=15)
testloader=torch.utils.data.DataLoader(test_dataset, batch_size=Bsize,shuffle=True,num_workers=15)

RegionProposer=YOLOr(saveprogress,
                     parameters=DEF_yolo,
                     groups=groups,
                     Testgroup=fold)

RegionProposer.train(trainloader,
                     testloader,
                     max_epochs=10,#30000,
                     patience=100,
                     max_time=12*60*60,
                     saveprogress=saveprogress,
                     savebest=savebest)

v,s = RegionProposer.false_pos_eval(testloader)
