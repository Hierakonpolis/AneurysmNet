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

import os, csv, warnings, itertools
import nibabel as nib
import numpy as np

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
        

def listdata(datafolder=dataroot):
    dataset={}
    
    for sample in os.scandir(datafolder):
        AddSampleFilePaths(sample.path,dataset)
        
    return dataset

