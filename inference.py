#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, torchvision
from models import Segmentation
import network as N
import nibabel as nib
import dataset as D

nets=['Unet_Inf_0_bes_dice.pth','Unet_Resized_bes_dice.pth'] # MAKE  THIS WORK!


torch.set_default_tensor_type('torch.FloatTensor') # t
torch.backends.cudnn.benchmark = True
Bsize=1
workers=0

tensor=D.ToTensor(order={'HD':3,'LD':3})
transforms= torchvision.transforms.Compose([tensor])
I=0

intof='TOF.nii.gz'
inmri='struct_aligned.nii.gz'
outfile='test.nii.gz'

intof='/input/pre/TOF.nii.gz'
inmri='/input/pre/struct_aligned.nii.gz'
outfile='/output/aneurysms.nii.gz'

outs,OrigSize=D.InResizer(intof,inmri)
MRInii, TOFnii = outs

for net in nets:
    Model=Segmentation(N.U_Net,
                        savefile=net,
                        parameters=None,
                        testset=None)
            
            

    dataset=D.OneVolPatchSet(TOFnii,MRInii,transforms)         
    trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, num_workers=workers)
    res=Model.inferece(trainloader,FinalThreshold=False,PreThreshold=True) ## CHECK THIS LINE LATER
    I=I+res/len(nets)

I[I>0.5]=1
I[I!=1 ]=0

out=D.OutResizer(I,OrigSize,TOFnii,order=0)
nib.save(out,outfile)