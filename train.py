#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, pickle, os, torchvision, sys
from models import Segmentation
import numpy as np
from network import DEF_PARAMS
import network as N
import dataset as D
from sklearn.model_selection import train_test_split

DEF_PARAMS['FilterSize']=3
DEF_PARAMS['FiltersNumHighRes']=np.array([64, 64, 64])
DEF_PARAMS['FiltersNumLowRes']=np.array([64, 64, 64])
DEF_PARAMS['FiltersDecoder']=np.array([64, 64, 64])
DEF_PARAMS['Categories']=int(3)
# DEF_PARAMS['Activation']=nn.LeakyReLU,
DEF_PARAMS['InblockSkip']=False
DEF_PARAMS['ResidualConnections']=True
DEF_PARAMS['PoolShape']=2
# DEF_PARAMS['BNorm']=nn.BatchNorm3d
# DEF_PARAMS['Conv']=nn.Conv3d
# DEF_PARAMS['Downsample']=PoolWrapper
# DEF_PARAMS['Upsample']=TransposeWrapper
DEF_PARAMS['InterpMode']='trilinear'
DEF_PARAMS['DownConvKernel']=3
DEF_PARAMS['Weights']=(0.001,1,0.5)
DEF_PARAMS['SideBranchWeight']=0.1
DEF_PARAMS['CCEweight']=1
DEF_PARAMS['DiceWeight']=1
DEF_PARAMS['WDecay']=0
DEF_PARAMS['TransposeSize']=4
DEF_PARAMS['TransposeStride']=2

dataroot='/media/Olowoo/ADAM_eqpatch'
datafile='databox[64 64 64].p'
saveroot='/media/Olowoo/ADAMsaves/'
name='U_Net_Final_res'

dataroot='/scratch/project_2003143/patches64_resized'
saveroot='/projappl/project_2003143'
if len(sys.argv)>1:
    dataroot=sys.argv[1]
    name=sys.argv[2]
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

tensor=D.ToTensor(order={'HD':3,'labels':0,'LD':3})

transforms= torchvision.transforms.Compose([tensor])
dataset=D.PatchesDataset(dataroot, datafile,transforms)



if os.path.isfile(saveprogress):
    Model=Segmentation(N.U_Net,
                   savefile=saveprogress,
                   parameters=DEF_PARAMS,
                   trainset=None,
                   testset=None)
    train_idxs, test_idxs = Model.opt['trainset'], Model.opt['testset']
else:
    

# shift=D.Shift([40,15,0],probability=0.5,order={'sample':3,'labels':0})
# rot=D.Rotate(5, probability=0.5,order={'sample':3,'labels':0})
# norm=D.Normalize(order={'sample':3,'labels':0})

    train_idxs, test_idxs = train_test_split(np.arange(len(dataset)), test_size=testsize)
    Model=Segmentation(N.U_Net,
                       savefile=None,
                       parameters=DEF_PARAMS,
                       trainset=train_idxs,
                       testset=test_idxs)


    

trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, sampler=torch.utils.data.SubsetRandomSampler(train_idxs),num_workers=workers)
testloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, sampler=torch.utils.data.SubsetRandomSampler(test_idxs),num_workers=workers)



Model.train(trainloader,
            testloader,
            max_epochs=MaxEpochs,
            patience=Patience,
            max_time=MaxTime,
            saveprogress=saveprogress,
            savebest=savebest)