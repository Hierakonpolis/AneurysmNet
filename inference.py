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

nets=['Unet_Inf_0_bes_dice.pth','Unet_Resized_bes_dice.pth']

# prefolder=sys.argv[1]
# outfolder=sys.argv[2]

torch.set_default_tensor_type('torch.FloatTensor') # t
torch.backends.cudnn.benchmark = True
Bsize=1
workers=8

tensor=D.ToTensor(order={'HD':3,'LD':3})
transforms= torchvision.transforms.Compose([tensor])
I=0

intof='/home/cat/ADAM_release_subjs/10072F/pre/TOF.nii.gz'
inmri='/home/cat/ADAM_release_subjs/10072F/pre/struct_aligned.nii.gz'
outs,OrigSize=D.InResizer(intof,inmri)
MRInii, TOFnii = outs

for net in nets:
    Model=Segmentation(N.U_Net,
                        savefile=net,
                        parameters=None,
                        testset=None)
            
            

    dataset=D.OneVolPatchSet(TOFnii,MRInii,transforms)         
    trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, num_workers=workers)
    res=Model.inferece(trainloader,FinalThreshold=True,PreThreshold=0.2)
    I=I+res/len(nets)
    
out=D.OutResizer(I,OrigSize,TOFnii,order=0)
nib.save(out,'/home/cat/ADAM_release_subjs/10072F/test.nii.gz')
       
import tqdm         
l=[]
for k in tqdm.tqdm(dataset):
    if k['L'].sum()>0:
        l.append(1)
    else:
        l.append(0)
#                 ref=nib.load(S+REF).get_fdata()
#                 ref[ref==2]=0
                
#                 allres.append(res.reshape([1]+list(res.shape)))
#                 allref.append(ref.reshape([1]+list(ref.shape)))
            
            
#             d0ice=[]
            
#             for thr in np.arange(0,1.05,0.05):
#                 for k in range(len(allres)):
#                     box=np.zeros_like(allres[k])
#                     box[allres[k]>=thr]=1
#                     d=SimpleDice(box, allref[k])
#                     d0ice.append(d)
#                 dices.append((np.round(np.mean(d0ice),4),np.round(thr,3),metric,net,PT,save))
#                 print(np.round(np.mean(d0ice),4),np.round(thr,3),metric,net,PT)
    
    