#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, torchvision
from models import Segmentation
import network as N
import nibabel as nib
import dataset as D

allnets=[['UnetRI0', 0.6049110487714825, 0.1],
         ['UnetRIGD0', 0.533970131792268, 0.45],
         ['CNNRIGD0', 0.47613174917380513, 0.45],
         ['UnetRI1', 0.6318399590042463, 0.1],
         ['UnetRIGD1', 0.5371896791864506, 0.85],
         ['CNNRIGD1', 0.3256386043593358, 0.8],
         ['UnetRI2', 0.4290913627380267, 0.5],
         ['UnetRIGD2', 0.5828611251139756, 0.3],
         ['CNNRIGD2', 0.14772716204486966, 0.55],
         ['UnetRI3', 0.5786958211017389, 0.35],
         ['UnetRIGD3', 0.6744822512317834, 0.2],
         ['CNNRIGD3', 0.43989162158605166, 0.85],
         ['UnetRI4', 0.5855979452755818, 0.1],
         ['UnetRIGD4', 0.6300422187087864, 0.3],
         ['CNNRIGD4', 0.19420637105835312, 0.4],
         ['UnetRI5', 0.3629931155754767, 0.25],
         ['UnetRIGD5', 0.49454304517251, 0.35],
         ['CNNRIGD5', 0.4123095695800509, 0.55]]


torch.set_default_tensor_type('torch.FloatTensor') # t
torch.backends.cudnn.benchmark = True
Bsize=1
workers=0#20

tensor=D.ToTensor(order={'HD':3,'LD':3})
transforms= torchvision.transforms.Compose([tensor])
I=0

intof='TOF.nii.gz'
inmri='struct_aligned.nii.gz'
outfile='test.nii.gz'
IN='/input'
outfile='/output/result.nii.gz'

# outfile='/home/cat/AneurysmNet/test.nii.gz'
# IN='/media/Olowoo/ADAM_release_subjs/10072F'

intof=IN+'/pre/TOF.nii.gz'
inmri=IN+'/pre/struct_aligned.nii.gz'

outs,OrigSize=D.InResizer(intof,inmri)
MRInii, TOFnii = outs

TW=0
for net, W, T in allnets:
    if 'CNN' in net:
        modtype=N.CNN
    else:
        modtype=N.U_Net
    print(net)
    TW+=W
    Model=Segmentation(modtype,
                        savefile=net+'_bes_dice.pth',
                        parameters=None,
                        testset=None)
    
    
    
    dataset=D.OneVolPatchSet(TOFnii,MRInii,transforms)
    trainloader=torch.utils.data.DataLoader(dataset, batch_size=Bsize, num_workers=workers)
    res=Model.inferece(trainloader,FinalThreshold=False,PreThreshold=True) ## CHECK THIS LINE LATER
    
    I=I+res*(0.5/T)*W

I=I/TW


# I[I>0.5]=1
# I[I!=1 ]=0

out=D.OutResizer(I,OrigSize,TOFnii,order=3,threshold=0.5)
nib.save(out,outfile)
