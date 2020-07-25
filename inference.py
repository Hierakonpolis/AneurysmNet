#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, pickle, os, torchvision, sys
from models import Segmentation
from Overlap import overlapping_patches
import numpy as np
from network import DEF_PARAMS
import network as N
import nibabel as nib
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

S='/media/Olowoo/ADAM_release_subjs/10054B'



def SimpleDice(A,B):
    return np.sum(2*A*B)/(A.sum()+B.sum())


# groupfile='groups.p'
addy='_res.nii.gz'
torch.set_default_tensor_type('torch.FloatTensor') # t
torch.backends.cudnn.benchmark = True
testsize=0.05
Bsize=10
workers=23
MaxEpochs=np.inf
Patience=10
MaxTime=np.inf

tensor=D.ToTensor(order={'HD':3,'LD':3})

transforms= torchvision.transforms.Compose([tensor])

base='/media/Olowoo/ADAMsaves/Unet'
second=['2C','_Dec_0001','_Dec_005','DecRes001','DR','_DW','_Inf_0','_Inf_dec001','_LF','_Resized']
third=['_bes_dice.pth','_best.pth','_prog.pth']
dices=[]
for net in second:
    for metric, file in zip(['Dice','Loss','Last'],third):
        print(net,metric)
        save=base+net+file
        if 'DR' or 'MF' in file:
            modeltype=N.U_Net_DR
        else:
            modeltype=N.U_Net
        
        Model=Segmentation(modeltype,
                       savefile=save,
                       parameters=DEF_PARAMS,
                       testset=None)
        
        allres=[]
        allref=[]
        for S1 in Model.opt['testset']:
            S=os.path.join('/media/Olowoo/ADAM_release_subjs',S1)
            
            TOF='/pre/TOF.nii.gz'
            STR='/pre/struct_aligned.nii.gz'
            REF='/aneurysms.nii.gz'
            if os.path.isfile(S+TOF+addy):
                TOF=TOF+addy
                STR=STR+addy
                REF=REF+addy
            dataset=D.OneVolPatchSet(S+TOF,S+STR,transforms)
        
            trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, num_workers=workers)
        
            res=Model.inferece(trainloader,FinalThreshold=False,PreThreshold=False)
            
            ref=nib.load(S+REF).get_fdata()
            ref[ref==2]=0
            
            allres.append(res.detach().cpu().numpy())
            allref.append(ref)
        allres=np.array(allres)
        allref=np.array(allref)
        for thr in np.arange(0.1,1,0.1):
            box=np.zeros_like(allres)
            box[allres>=thr]=1
            d=SimpleDice(box, allref)
            dices.append((d,thr,metric,net))
    