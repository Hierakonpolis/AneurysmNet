#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, pickle, os, torchvision, sys
from models import Segmentation
import numpy as np
from network import CascadedDecoder,  DEF_DEEPLAB
import network as N
import dataset as D
from sklearn.model_selection import train_test_split
from DeepLab.models.DeepLab import DeepLabv3Plus

dataroot='/media/Olowoo/patches_redux'
datafile='databox[64 64 64].p'
saveroot='/media/Olowoo/ADAMsaves/'
name='DeepLab_Upconv'

dataroot='/scratch/project_2003143/patches64'
saveroot='/projappl/project_2003143'
if len(sys.argv)>1:
    dataroot=sys.argv[1]
# groupfile='groups.p'

saveprogress=os.path.join(saveroot,name+'_prog.pth')
savebest=os.path.join(saveroot,name+'_best.pth')
torch.set_default_tensor_type('torch.cuda.FloatTensor') # t
torch.backends.cudnn.benchmark = True
testsize=0.05
Bsize=8
workers=19
MaxEpochs=np.inf
Patience=10
MaxTime=np.inf

# shift=D.Shift([40,15,0],probability=0.5,order={'sample':3,'labels':0})
# rot=D.Rotate(5, probability=0.5,order={'sample':3,'labels':0})
# norm=D.Normalize(order={'sample':3,'labels':0})
tensor=D.ToTensor(order={'HD':3,'labels':0,'LD':3})

transforms= torchvision.transforms.Compose([tensor])

dataset=D.PatchesDataset(dataroot, datafile,transforms)

train_idxs, test_idxs = train_test_split(np.arange(len(dataset)), test_size=testsize)
trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, sampler=torch.utils.data.SubsetRandomSampler(train_idxs),num_workers=workers)
testloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, sampler=torch.utils.data.SubsetRandomSampler(test_idxs),num_workers=workers)

Model=Segmentation(DeepLabv3Plus,
                   savefile=None,
                   parameters=DEF_DEEPLAB,
                   trainset=train_idxs,
                   testset=test_idxs)

Model.train(trainloader,
            testloader,
            max_epochs=MaxEpochs,
            patience=Patience,
            max_time=MaxTime,
            saveprogress=saveprogress,
            savebest=savebest)
