#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, pickle, os, torchvision, sys
from models import Segmentation
from Overlap import overlapping_patches
import numpy as np
from network import DEF_PARAMS
import network as N
import nibabel as nib
import matplotlib.pyplot as plt
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
second=['2C','_Dec_0001','_Dec_005','DecRes001','_DW','_Inf_0','_Inf_dec001','_LF','_Resized','DR']
second=['AllLosses','DRV']
third=['_bes_dice.pth','_best.pth','_prog.pth']
dices=pickle.load(open('/media/Olowoo/ADAMsaves/saveres.p','rb'))
Prehreshold=[True,False]
results={}

# UnetDRV resized
#AllLosses resized
# AllDR resized
# Simp resized
# Unet32 original
# UnetInterpRes resized
# UnetStrideRes resized




for net in second:
    for metric, file in zip(['Dice','Loss','Last'],third):
        for PT in Prehreshold:
            print(net,metric)
            save=base+net+file
            if 'DR' or 'MF' in file:
                modeltype=N.U_Net_DR
            else:
                modeltype=N.U_Net
            if '32' in file:
                BS=32
            else:
                BS=64
            
            Model=Segmentation(modeltype,
                            savefile=save,
                            parameters=None,
                            testset=None)
            
            l=[x[1] for x in Model.opt['TestLoss']]
            plt.plot(l)
            plt.plot(Model.opt['TrainingLoss'])
            plt.title(net+file)
            plt.show()
            allres=[]
            allref=[]

            for S1 in Model.opt['testset']:
                S=os.path.join('/media/Olowoo/ADAM_release_subjs',S1)
                
                TOF='/pre/TOF.nii.gz'
                STR='/pre/struct_aligned.nii.gz'
                REF='/aneurysms.nii.gz'
                # if 'Res' in net:
                TOF=TOF+'_standard.nii.gz'
                STR=STR+'_standard.nii.gz'
                REF=REF+'_standard.nii.gz'
                    # print('!')
                # if os.path.isfile(S+TOF+addy):
                #     TOF=TOF+addy
                #     STR=STR+addy
                #     REF=REF+addy
                dataset=D.OneVolPatchSet(S+TOF,S+STR,transforms,box_size=BS)
            
                trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, num_workers=workers)
            
            
            
                res=Model.inferece(trainloader,FinalThreshold=False,PreThreshold=PT)
                
                ref=nib.load(S+REF).get_fdata()
                ref[ref==2]=0
                
                allres.append(res.reshape([1]+list(res.shape)))
                allref.append(ref.reshape([1]+list(ref.shape)))
            
            
            d0ice=[]
            
            for thr in np.arange(0,1.05,0.05):
                for k in range(len(allres)):
                    box=np.zeros_like(allres[k])
                    box[allres[k]>=thr]=1
                    d=SimpleDice(box, allref[k])
                    d0ice.append(d)
                dices.append((np.round(np.mean(d0ice),4),np.round(thr,3),metric,net,PT,save))
                print(np.round(np.mean(d0ice),4),np.round(thr,3),metric,net,PT)
                
pickle.dump(dices,open('/media/Olowoo/ADAMsaves/saveres.p','wb'))

name=''
M=0
K=''
for k in dices:
    if name!=k[3]: 
        print(K)
        name=k[3]
        M=0
        
    if k[0]>M:
        M=k[0]
        K=k
print(K)
name=k[3]
M=0

    
    
    