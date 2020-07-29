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
DEF_PARAMS['SobelWeight']=1
DEF_PARAMS['SurfaceLossWeight']=1
            
dataroot='/media/Olowoo/ADAM_eqpatch'
datafile='databox[64 64 64].p'
saveroot='/tmp/ss'
name='U_677_res'

# dataroot='/scratch/project_2003143/patches64_resized'
# saveroot='/projappl/project_2003143'
fold=0

if len(sys.argv)>1:
    dataroot=sys.argv[1]
    name=sys.argv[2]
    fold=int(sys.argv[3])
# groupfile='groups.p'

folds={0:['10078F', '10042', '10072F', '10031', '10026'],
       1:['10062B', '10045B', '10071F', '10078F', '10010'],
       2:['10051B', '10070B', '10013', '10057B', '10076B'],
       3:['10061F', '10003', '10076B', '10057B', '10065F'],
       4:['10048F', '10047F', '10015', '10066B', '10016'],
       5:['10070B', '10076B', '10076F', '10042', '10037']}

fold=folds[fold]


saveprogress=os.path.join(saveroot,name+'_prog.pth')
savebest=os.path.join(saveroot,name+'_best.pth')
torch.set_default_tensor_type('torch.cuda.FloatTensor') # t
torch.backends.cudnn.benchmark = True
testsize=0.05
Bsize=8
workers=19
MaxEpochs=np.inf
Patience=np.inf
MaxTime=np.inf

tensor=D.ToTensor(order={'HD':3,'labels':0,'LD':3})

transforms= torchvision.transforms.Compose([tensor])

trainset=D.PatchesDataset(dataroot, datafile,transforms,fold,False)
testset=D.PatchesDataset(dataroot, datafile,transforms,fold,True)



if os.path.isfile(saveprogress):
    Model=Segmentation(N.U_Net,
                   savefile=saveprogress,
                   parameters=DEF_PARAMS,
                   testset=None)
else:
    

# shift=D.Shift([40,15,0],probability=0.5,order={'sample':3,'labels':0})
# rot=D.Rotate(5, probability=0.5,order={'sample':3,'labels':0})
# norm=D.Normalize(order={'sample':3,'labels':0})

    Model=Segmentation(N.U_Net,
                       savefile=None,
                       parameters=DEF_PARAMS,
                       testset=fold)


    

trainloader=torch.utils.data.DataLoader(trainset, batch_size=Bsize, shuffle=True,num_workers=workers)
testloader=torch.utils.data.DataLoader(testset, batch_size=Bsize, shuffle=True,num_workers=workers)



Model.train(trainloader,
            testloader,
            max_epochs=MaxEpochs,
            patience=Patience,
            max_time=MaxTime,
            saveprogress=saveprogress,
            savebest=savebest)
