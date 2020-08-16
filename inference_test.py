#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, pickle, os, torchvision, sys
from models import Segmentation, Rebuild
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
    return (np.sum(2*A*B)+1e-10)/(A.sum()+B.sum()+1e-10)


# groupfile='groups.p'
addy='_res.nii.gz'
torch.set_default_tensor_type('torch.FloatTensor') # t
torch.backends.cudnn.benchmark = True
testsize=0.05
Bsize=5
workers=23
MaxEpochs=np.inf
Patience=10
MaxTime=np.inf

tensor=D.ToTensor(order={'HD':3,'LD':3})

transforms= torchvision.transforms.Compose([tensor])

base='/media/Olowoo/ADAMsaves/'
# second=['2C','_Dec_0001','_Dec_005','DecRes001','_DW','_Inf_0','_Inf_dec001','_LF','_Resized','DR']
second=['UnetRI1','UnetRIGD1','CNNRIGD1']
third=['_bes_dice.pth']#,'_best.pth']
dices=pickle.load(open('/media/Olowoo/ADAMsaves/saveres.p','rb'))
Prehreshold=[True]#,False]
results={}

# UnetDRV resized
#AllLosses resized
# AllDR resized
# Simp resized
# Unet32 original
# UnetInterpRes resized
# UnetStrideRes resized



for NN in [0,1,2,3,4,5]:
    Nn=str(NN)
    RN=pickle.load(open('R'+str(NN)+'.p','rb'))
    second=['UnetRIGD'+Nn+'a']
    for net in second:
        refs=[]
        ress=[]
        for metric, file in zip(['Dice'],third):
            for PT in Prehreshold:
                
                print(net,metric)
                save=base+net+file
                if ('DR' in net) or ('MF' in net):
                    modeltype=N.U_Net_DR
                    print('DR')
                elif 'Simple' in net:
                    modeltype=N.U_Net_Single
                else:
                    modeltype=N.U_Net
                if '32' in net:
                    BS=32
                    print('32')
                else:
                    BS=64
                
                if 'CNN' in net:
                    modeltype=N.CNN
                else:
                    modeltype=N.U_Net
                
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
                    if '32' not in net:
                        TOF=TOF+'_standard.nii.gz'
                        STR=STR+'_standard.nii.gz'
                        REF=REF+'_standard.nii.gz'
                        print('!')
                    if os.path.isfile(S+TOF+addy):
                        TOF=TOF+addy
                        STR=STR+addy
                        REF=REF+addy
                    dataset=D.OneVolPatchSet(S+TOF,S+STR,transforms,box_size=BS)
                
                    trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, num_workers=workers)
                
                
                
                    res=Model.inferece(trainloader,FinalThreshold=False,PreThreshold=PT)
                    
                    ress=res
                    
                    ref=nib.load(S+REF).get_fdata()
                    ref[ref==2]=0
                    refs.append(ref)
                    allres.append(res.reshape([1]+list(res.shape)))
                    allref.append(ref.reshape([1]+list(ref.shape)))
                
                RN[net]=allres
    pickle.dump(RN,open('R'+Nn+'.p','wb'))
                
            # d0ice=[]
            
            # for thr in np.arange(0,1.05,0.05):
            #     for k in range(len(allres)):
            #         box=np.zeros_like(allres[k])
            #         box[allres[k]>=thr]=1
            #         d=SimpleDice(box, allref[k])
            #         d0ice.append(d)
            #     dices.append((np.round(np.mean(d0ice),4),np.round(thr,3),metric,net,PT,save))
            #     print(np.round(np.mean(d0ice),4),np.round(thr,3),metric,net,PT)
                
# pickle.dump(dices,open('/media/Olowoo/ADAMsaves/saveres.p','wb'))

# name=''
# M=0
# K=''
# for k in dices:
#     if name!=k[3]: 
#         print(K)
#         name=k[3]
#         M=0
        
#     if k[0]>M:
#         M=k[0]
#         K=k
# print(K)
# name=k[3]
# M=0


# save=base+'_Resized'+'_prog.pth'
# Model=Segmentation(N.U_Net,
#                     savefile=save,
#                     parameters=None,
#                     testset=None)
# S=os.path.join('/media/Olowoo/ADAM_release_subjs','10078F')
# dataset=D.OneVolPatchSet(S+TOF,S+STR,transforms,box_size=64)
# trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, num_workers=workers)
# patches,centerpoints=Model.inferece(trainloader,FinalThreshold=False,PreThreshold=False)
# ref=nib.load(S+REF)
# reff=ref.get_fdata()
# truepatches=[]

# for k in centerpoints:
#     low, high, _ = D.Patch(dataset.vol,64,None,k)
#     truepatches.append(D.Carve(low,high,reff))
# dices=[]
# Ts=[]
# predicc=[]
# center=[]
# CCs=[]
# for k in range(len(centerpoints)):
#     b=0
#     for thr in list(np.arange(0,1,0.05))+[0.001,0.01,0.0001]:
#         box=np.zeros_like(patches[k])
#         box[patches[k]>=thr]=1
#         d=SimpleDice(box, truepatches[k])
        
#         if d>b: 
#             b=d
#             BT=thr
#     # if np.sum(truepatches[k])>0:
#     dices.append(b)
#     Ts.append(BT)
#     box=np.zeros_like(patches[k])
#     box[patches[k]>=BT]=1
#     d=SimpleDice(box, truepatches[k])
#     assert d==b
#     predicc.append(np.copy(box))
#     CCs.append(D.label(box).max())
#     center.append(centerpoints[k])
        
        
# dices=np.array(dices)
# Ts=np.array(Ts)
# CCs=np.array(CCs)

# out=Rebuild(dataset.vol, predicc, center)
# z=np.zeros_like(out)
# z[out>0.5]=1
# print(SimpleDice(z, reff))
# # 
# # a=np.zeros_like(res)
# # a[res>0.0001]=1
# # l=D.label(a)
# # lm=l.max()
# # nib.save(nib.Nifti1Image(a,ref.affine),'/media/Olowoo/ADAM_release_subjs/10078F/test.nii.gz')
# # print(SimpleDice(a,ref.get_fdata()))

##############################################################################
# weights=[1,1,1]#[0.567,0.509,0.469]
# weightsF1=[0.63,0.35,0.319]
# thresholdsF1=[0.05,0.35,0.35]
# thresholds= [0.5,0.5,0.5]#[0.1, 0.45, 0.45]

# for thr in list(np.arange(0,1,0.05)):
#     d0ice=[]
#     for sample in range(5):
#         wsum=0
#         sambox=[]
#         for net, W , T in zip(R.keys(),weightsF1,thresholdsF1):
#             if net in 'UnetRI0': 
#                 sambox.append(R[net][sample]*W*(0.5/T))
#                 wsum+=W
#         sambox=np.concatenate(sambox,axis=0)
#         sambox=np.sum(sambox,axis=0)/np.sum(wsum)
#         sambox[sambox>thr]=1
#         sambox[sambox<1]=0
#         d0ice.append(SimpleDice(sambox,refs[sample]))
#     print(np.mean(d0ice),thr)

everyone=[]

for NN in range(6):
    R=pickle.load(open('R'+str(NN)+'.p','rb'))
    Model=Segmentation(N.U_Net,
                                savefile='/media/Olowoo/ADAMsaves/UnetRIGD'+str(NN)+'_bes_dice.pth',
                                parameters=None,
                                testset=None)
    refs=[]
    
    for S1 in Model.opt['testset']:
                    S=os.path.join('/media/Olowoo/ADAM_release_subjs',S1)
                    ref=nib.load(S+REF).get_fdata()
                    ref[ref>1.5]=0
                    ref[ref>0.5]=1
                    ref[ref<1]=0
                    refs.append(ref.reshape([1]+list(ref.shape)))
                    
    for net in R:
        bestdice=0
        for thr in list(np.arange(0,1,0.05)):
            doice=[]
            for sample in range(5):
                box=np.copy(R[net][sample])
                box[box>thr]=1
                box[box<1]=0
                
                doice.append(SimpleDice(box,refs[sample]))
            dice=np.mean(doice)
            if dice>bestdice:
                bestdice=dice
                T=thr
        everyone.append([net,bestdice,T])
        print([net,bestdice,T])
pickle.dump(everyone,open('everynet.p','wb'))
##############################################################################
# pickle.dump(R1,open('R1.p','wb'))
# pickle.dump(refs,open('refs.p','wb'))
        