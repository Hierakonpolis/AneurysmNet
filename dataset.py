#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset wrappers, transforms and related funcions.


a text file with the 3D voxel coordinates of the centre of mass of the aneurysms and the maximum radius of the aneurysm
a label image with labels:
0 = Background
1 = Untreated, unruptured aneurysm
2 = Treated aneurysms or artefacts resulting from treated aneurysms

"""

import os, csv, warnings, itertools, torch
import nibabel as nib
from scipy.ndimage.interpolation import shift
from scipy.ndimage import rotate
import numpy as np
from torch.utils.data import Dataset
torch.set_default_tensor_type('torch.cuda.FloatTensor') 

dataroot='/media/Olowoo/ADAM_release_subjs'

def AddSampleFilePaths(folder,foldersdict):
    
    dataset=foldersdict
    sample=folder.rstrip('/').rstrip('\\')
    sample=os.path.basename(sample)
    
    dataset[sample]={}
    locations_file=os.path.join(folder,'location.txt')
    dataset[sample]['locations']=[]
    
    with open(locations_file,'r') as f:
        reader=csv.reader(f)
        for row in reader: 
            dataset[sample]['locations'].append(row)
    
    dataset[sample]['label']=os.path.join(folder, 'aneurysms.nii.gz')
    dataset[sample]['TOF orig']=os.path.join(folder,'orig', 'TOF.nii.gz')
    dataset[sample]['TOF']=os.path.join(folder,'pre', 'TOF.nii.gz')
    dataset[sample]['struct orig']=os.path.join(folder,'orig', 'struct.nii.gz')
    dataset[sample]['struct']=os.path.join(folder,'pre', 'struct.nii.gz')
    dataset[sample]['ID']=sample


def AddToYolo(SamplePaths):
    MRI=nib.load(SamplePaths['struct']).get_fdata()
    TOF=nib.load(SamplePaths['TOF']).get_fdata()
    LAB=nib.load(SamplePaths['label']).get_fdata()
    
    ID = SamplePaths['ID']
    
    shape=[1]+list(MRI.shape)
    MRI=MRI.reshape(shape)
    TOF=TOF.reshape(shape)
    LAB[LAB!=0]=1
    LAB=LAB.reshape(shape)
    
    IN = np.concatenate([MRI,TOF],axis=0)
    
    out={'ID':ID,
         'sample':IN,
         'labels':LAB}
    
    return out


def listdata(datafolder=dataroot):
    dataset={}
    
    for sample in os.scandir(datafolder):
        AddSampleFilePaths(sample.path,dataset)
        
    return dataset

class YoloDataset(Dataset):
    def __init__(self,datafolder=dataroot,transforms=None):
        samples=listdata(datafolder)
        self.transforms=transforms
        self.dataset=[]
        
        for sample in samples:
            self.dataset.append(sample)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        sample=self.dataset[idx]
        
        if self.transforms:
            sample=self.transforms(sample)
        
        return sample
        
class Shift():
    """
    
    Shifts volume by a random amount ranging between +- max_shift, depending on
    axis, with probability=probability. Specify which labels and order of
    interpolation in the order dictionary
    
    """
    def __init__(self,max_shift=[0,0,0],probability=0.5,order={'sample':3,'labels':0}):
        self.order=order
        self.maxshift=max_shift
        self.probability=probability
        
    def __call__(self,sample):
        if float(np.random.random(1))<=self.probability:
            
            delta=[0]+list(np.random.uniform(-1,1,size=3)*self.maxshift)
            
            for vol, order in self.order.items():
                sample[vol]=shift(sample[vol],delta,order=order)
            
        
        return sample

class Rotate():
    def __init__(self,angle,probability,order={'sample':3,'labels':0}):
        self.order=order
        self.angle=angle
        self.probability=probability
        
    def __call__(self,sample):
        if float(np.random.random(1))<= self.probability:
            
            Ang=float(np.random.uniform(-self.MaxAngle,self.MaxAngle))
            
            for vol, order in self.order.items():
                shape=1+list(sample[vol].shape[-3:])
                vols=np.split(sample[vol],sample[vol].shape[0],axis=0)
                vols= [rotate(k.reshape(shape[-3:]),Ang,axes=(0,1),reshape=False,order=order).reshape(shape) for k in vols]
                
                sample[vol]=np.concatenate(vols,axis=0)
                
        return sample

def normalize(X):
    return (X-X.mean())/X.std()

class Normalize():
    def __init__(self,order={'sample':3,'labels':0}):
        self.order=order
    
    def __call__(self,sample):
        for vol,order in self.order.items():
            if order !=0:
                shape=1+list(sample[vol].shape[-3:])
                vols=np.split(sample[vol],sample[vol].shape[0],axis=0)
                vols=[normalize(vol).reshape(shape) for vol in vols]
                
                sample[vol]=np.concatenate(vols,axis=0)
        
        return sample
    
class ToTensor():
    def __init__(self,device='cuda',order={'sample':3,'labels':0}):
        self.device=device
        self.order=order
    def __call__(self,sample):
        for key in self.order:
            sample[key]=torch.from_numpy(sample[key]).float().to(self.device)
    
        
        