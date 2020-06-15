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

import os, csv, warnings, itertools, torch, shutil, tqdm, pickle
import nibabel as nib
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate
import numpy as np
from skimage.measure import label
from torch.utils.data import Dataset
torch.set_default_tensor_type('torch.cuda.FloatTensor') 

def IDListMaker(groups,which,mode='train'):
    if mode=='test':
        return groups[which]
    else:
        L=[]
        for k in groups:
            if k !=which: L=L+groups[k]
    return L
        

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
    dataset[sample]['struct orig']=os.path.join(folder,'orig', 'struct_aligned.nii.gz')
    dataset[sample]['struct']=os.path.join(folder,'pre', 'struct_aligned.nii.gz')
    dataset[sample]['ID']=sample



def MakePatches(dataroot,destination):
    pass


def AddToYolo(SamplePaths):
    
    if os.path.isfile(SamplePaths['struct']+'_res.nii.gz'):
        MRI=nib.load(SamplePaths['struct']+'_res.nii.gz').get_fdata()
        TOF=nib.load(SamplePaths['TOF']+'_res.nii.gz').get_fdata()
        LAB=nib.load(SamplePaths['label']+'_res.nii.gz').get_fdata()
    else:
        
        MRI=nib.load(SamplePaths['struct']).get_fdata()
        TOF=nib.load(SamplePaths['TOF']).get_fdata()
        LAB=nib.load(SamplePaths['label']).get_fdata()
    
    ID = SamplePaths['ID']
    
    shape=list(MRI.shape)
    
    xg, yg, zg = np.meshgrid(range(shape[0]),range(shape[1]),range(shape[2]),indexing='ij')
    xg=xg/shape[0]
    yg=yg/shape[1]
    zg=zg/shape[2]
    
    shape=[1]+shape
    
    coords=np.concatenate((xg.reshape(shape), yg.reshape(shape), zg.reshape(shape)),axis=0)
    
    MRI=MRI.reshape(shape)
    TOF=TOF.reshape(shape)
    
    LAB[LAB!=0]=1
    LAB=LAB.reshape(shape)
    
    IN = np.concatenate([MRI,TOF],axis=0)
    
    out={'ID':ID,
         'coords':coords,
         'sample':IN,
         'labels':LAB}
    
    return out

def Carve(low,high,vol):
    
    return vol[low[0]:high[0],low[1]:high[1],low[2]:high[2]]

def Patch(connected_components,size,positive,loc=None):
    if loc is None:
        if positive:
            x,y,z = np.where(connected_components>0)
            num=connected_components.max()
            bignesses=[len(connected_components[connected_components==k]) for k in range(1,num+1)]
            weights=np.zeros_like(connected_components).astype(float)
            
            for bigness, index in zip(bignesses, range(1,num+1)):
                weights[connected_components==index]=1/bigness*float(min(bignesses))
            
            weights=[weights[x[k],y[k],z[k]] for k in range(len(x))]
            weights=np.array(weights)
            
            loc=np.random.choice(np.arange(0,len(x),1),p=weights/weights.sum())
            loc=np.array((x[loc],y[loc],z[loc]))
            
            loc=loc+(np.random.random(3)-0.5)*size
        else:
            loc=np.array(connected_components.shape)-size
            loc=loc*np.random.random(3)+size/2
    
    loc=np.round(loc)
    maxs=np.array(connected_components.shape)-1
    lows=np.clip(loc-np.array(size/2).astype(int),0,maxs).astype(int)
    highs=np.clip(loc+np.array(size/2).astype(int),0,maxs).astype(int)
    
    highs[lows==0]=size[lows==0]
    
    maxmins=maxs-size
    lows[highs==maxs]=maxmins[highs==maxs]
    
    return lows, highs, loc

def GetPosFrac(dataroot):
    samples=listdata(dataroot).values()
    pos=0
    for sample in samples:
        LAB=nib.load(sample['label']).get_fdata()
        if np.sum(LAB)>0:
            pos+=1
    return pos/len(samples)


def ExtractPatches(dataroot,outfolder,number,size=64):
    si=str(size)
    samples=listdata(dataroot)
    assert type(size)==int or len(size)==3
    size=np.array((1,1,1))*np.array(size)
    datalist=[]
    outlist=os.path.join(outfolder,'databox'+str(size)+'.p')
    if os.path.isfile(outlist): datalist=pickle.load(open(outlist,'rb'))
    
    
    pos_frac=0.8230088495575221
    
    for sample in samples.values():
        
        
        if os.path.isfile(sample['struct']+'_res.nii.gz'):
            MRI=nib.load(sample['struct']+'_res.nii.gz').get_fdata()
            TOF=nib.load(sample['TOF']+'_res.nii.gz')
            LAB=nib.load(sample['label']+'_res.nii.gz').get_fdata()
        else:
            
            MRI=nib.load(sample['struct']).get_fdata()
            TOF=nib.load(sample['TOF'])
            LAB=nib.load(sample['label']).get_fdata()
        
        hea=TOF.header
        aff=TOF.affine
        LAB=np.round(LAB)
        TOF=normalize(TOF.get_fdata())
        MRI=normalize(MRI)
        
        connected_components=label(LAB)
        
        if np.any(LAB>0): 
            HasPos=True
        else:
            HasPos=False
        
        numpos=int(number/pos_frac)
        numneg=int(number)
        
        if not os.path.isdir(os.path.join(outfolder, sample['ID'])): os.mkdir(os.path.join(outfolder, sample['ID']))
        
        for pos, nametag, num in zip ((True,False),('pos','neg'),(numpos,numneg)):
            
            if (not HasPos) and pos: continue
        
            for n in range(num):
                
                if os.path.isfile(os.path.join(outfolder, sample['ID'],'_STR_'+str(n)+'_'+si+'_'+nametag+'.nii.gz')): continue
                S={'ID':sample['ID'],
                   'struct':os.path.join( sample['ID'],'STR_'+str(n)+'_'+si+'_'+nametag+'.nii.gz'),
                   'TOF':os.path.join( sample['ID'],'TOF_'+str(n)+'_'+si+'_'+nametag+'.nii.gz'),
                   'labels':os.path.join( sample['ID'],'ane_'+str(n)+'_'+si+'_'+nametag+'.nii.gz'),
                   
                   'structB':os.path.join( sample['ID'],'STR_'+str(n)+'_'+si+'_big_'+nametag+'.nii.gz'),
                   'TOFB':os.path.join( sample['ID'],'TOF_'+str(n)+'_'+si+'_big_'+nametag+'.nii.gz'),
                   'labelsB':os.path.join( sample['ID'],'ane_'+str(n)+'_'+si+'_big_'+nametag+'.nii.gz'),
                   'type':'Positive'}
                    
            
                low, high, ref = Patch(connected_components,size,pos)
                lab=Carve(low,high,LAB)
                mri=Carve(low,high,MRI)
                tof=Carve(low,high,TOF)
                
                low, high, _ = Patch(connected_components,size*2,pos,ref)
                labB=Carve(low,high,LAB)
                mriB=Carve(low,high,MRI)
                tofB=Carve(low,high,TOF)
                
                
                zf=np.array(lab.shape)/np.array(labB.shape)
                
                labB=zoom(labB,zf,order=0)
                mriB=zoom(mriB,zf,order=3)
                tofB=zoom(tofB,zf,order=3)
                
                assert labB.shape==lab.shape
                
                nib.save(nib.Nifti1Image(lab, aff, hea),os.path.join(outfolder, S['labels']))
                nib.save(nib.Nifti1Image(mri, aff, hea),os.path.join(outfolder, S['struct']))
                nib.save(nib.Nifti1Image(tof, aff, hea),os.path.join(outfolder, S['TOF']))
                
                nib.save(nib.Nifti1Image(labB, aff, hea),os.path.join(outfolder, S['labelsB']))
                nib.save(nib.Nifti1Image(mriB, aff, hea),os.path.join(outfolder, S['structB']))
                nib.save(nib.Nifti1Image(tofB, aff, hea),os.path.join(outfolder, S['TOFB']))
                datalist.append(S)
                
                
            datalist.append(S)
    pickle.dump(datalist,open(outlist,'wb'))
    
def assign(LAB,value):
    a=np.zeros_like(LAB)
    a[LAB==value]=1
    return a

def AddPatch(sampledict,root):
    print(sampledict)
    print(root)
    out={}
    LAB=nib.load(os.path.join(root,sampledict['labels'])).get_fdata()
    shape=[1]+list(LAB.shape)
    print(shape)
    print(LAB.shape)
    labels=np.zeros([3]+list(LAB.shape))
    print(labels.shape)
    
    
    # labels[0,:,:,:][LAB==0]=1
    # labels[1,:,:,:][LAB==1]=1
    # labels[2,:,:,:][LAB==2]=1
    
    labels[0,:,:,:]=assign(LAB,0)
    labels[1,:,:,:]=assign(LAB,1)
    labels[2,:,:,:]=assign(LAB,2)
    
    MRI_HD=nib.load(os.path.join(root,sampledict['struct'])).get_fdata().reshape(shape)
    MRI_LD=nib.load(os.path.join(root,sampledict['structB'])).get_fdata().reshape(shape)
    
    TOF_HD=nib.load(os.path.join(root,sampledict['TOF'])).get_fdata().reshape(shape)
    TOF_LD=nib.load(os.path.join(root,sampledict['TOFB'])).get_fdata().reshape(shape)
    
    out['HD']=np.concatenate([MRI_HD,TOF_HD],axis=0)
    out['LD']=np.concatenate([MRI_LD,TOF_LD],axis=0)
    
    out['labels']=labels
    
    
    return out
    


def YourFriendlyResizer(datapath,standardsize=560):
    """
    560 scelto per pixdim: 1024*0.1953125/0.35714287

    """
    dataset=listdata(datapath)
    for paths in tqdm.tqdm(dataset.values()):
            
        MRI=nib.load(paths['struct'])
        TOF=nib.load(paths['TOF'])
        LAB=nib.load(paths['label'])
        nibs=[MRI,TOF,LAB]
        ps=[paths['struct'],paths['TOF'],paths['label']]
        ords=[3,3,0]
        if MRI.shape[0]>=600:
            for ni, o, p in zip(nibs,ords,ps):
                assert MRI.shape[1] == MRI.shape[0]
                shapefactor=[standardsize/MRI.shape[0],standardsize/MRI.shape[1],1]
                newvol=zoom(ni.get_fdata(),shapefactor,order=o)
                H=ni.header
                H['dim'][1]=standardsize
                H['dim'][2]=standardsize
                H['pixdim'][1]=H['pixdim'][1]*standardsize/MRI.shape[0]
                H['pixdim'][2]=H['pixdim'][2]*standardsize/MRI.shape[1]
                newnii=nib.Nifti1Image(newvol, ni.affine,header=ni.header)
                nib.save(newnii, p+'_res.nii.gz')
    

def listdata(datafolder):
    dataset={}
    
    for sample in os.scandir(datafolder):
        AddSampleFilePaths(sample.path,dataset)
        
    return dataset

class YoloDataset(Dataset):
    def __init__(self,datafolder,transforms=None,IDs=None):
        samples=listdata(datafolder)
        self.transforms=transforms
        self.dataset=[]
        
        for ID, sample in samples.items():
            
            if IDs and (ID not in IDs): continue
        
            self.dataset.append(sample)
    
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        sample=AddToYolo(self.dataset[idx])
        
        if self.transforms:
            sample=self.transforms(sample)
        
        return sample

class PatchesDataset(Dataset):
    def __init__(self,patchesroot,databoxfile,transforms=None):
        self.dataset=pickle.load(open(os.path.join(patchesroot,databoxfile),'rb'))
        self.path=patchesroot
        self.transforms=transforms
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        #print(self.dataset[idx])
        #print(idx)
        #print(self.path)
        
        sample=AddPatch(self.dataset[idx],self.path)
        
        if self.transforms:
            sample=self.transforms(sample)
        
        return sample
        
        

class Shift():
    """
    
    Shifts volume by a random amount ranging between +- max_shift, depending on
    axis, with probability=probability. Specify which labels and order of
    interpolation in the order dictionary
    
    """
    def __init__(self,max_shift=[0,0,0],probability=0.5,order={'sample':3,'labels':0},AlwaysApply=False):
        self.order=order
        self.maxshift=max_shift
        self.probability=probability
        self.AlwaysApply=AlwaysApply
        
    def __call__(self,sample):
        approve=sample['sample'].shape[0]>=500 or self.AlwaysApply
        
        if float(np.random.random(1))<=self.probability and approve:
            
            delta=[0]+list(np.random.uniform(-1,1,size=3)*self.maxshift)
            
            for vol, order in self.order.items():
                sample[vol]=shift(sample[vol],delta,order=order)
            
        
        return sample

class Rotate():
    def __init__(self,angle,probability,order={'sample':3,'labels':0},AlwaysApply=False):
        self.order=order
        self.angle=angle
        self.AlwaysApply=AlwaysApply
        self.probability=probability
        
    def __call__(self,sample):
        approve=sample['sample'].shape[0]>=500 or self.AlwaysApply
        
        if float(np.random.random(1))<= self.probability and approve:
            
            Ang=float(np.random.uniform(-self.angle,self.angle))
            
            for vol, order in self.order.items():
                shape=[1]+list(sample[vol].shape[-3:])
                vols = np.split(sample[vol],sample[vol].shape[0],axis=0)
                vols = [rotate(k.reshape(shape[-3:]),Ang,axes=(0,1),reshape=False,order=order).reshape(shape) for k in vols]
                
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
                shape=[1]+list(sample[vol].shape[-3:])
                vols=np.split(sample[vol],sample[vol].shape[0],axis=0)
                vols=[normalize(vol).reshape(shape) for vol in vols]
                
                sample[vol]=np.concatenate(vols,axis=0)
        
        return sample


class ToTensor():
    def __init__(self,device='cuda',order={'sample':3,'labels':0}):
        self.device=device
        self.order=order
    def __call__(self,sample):
        # print('ToTensor')
        for key in self.order:
            sample[key]=torch.from_numpy(sample[key]).float()#.to(self.device)
            
        
        return sample
    
        
        
