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

import os, csv, torch, tqdm, pickle
import nibabel as nib
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate
import numpy as np
from skimage.measure import label
from torch.utils.data import Dataset
from Faster import potential_aneurysm
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



class OneVolPatchSet(Dataset):
    def __init__(self,volumepath,anatomy,transforms=None,Labels=None):
        
        if type(volumepath) == str:
            v=nib.load(volumepath).get_fdata()
            a=nib.load(anatomy).get_fdata()
        elif type(volumepath) == nib.nifti1.Nifti1Image:
            v=volumepath.get_fdata()
            a=anatomy.get_fdata()
        self.vol=normalize(v)
        self.anatomy=normalize(a)
        self.locations=potential_aneurysm(self.vol)
        self.GetLabels=Labels
        if type(Labels) == str:
            self.GT=nib.load(Labels).get_fdata()
        
        # self.dataset=pickle.load(open(os.path.join(patchesroot,databoxfile),'rb'))
        # self.path=patchesroot
        self.transforms=transforms
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self,idx):
        low, high, loc =Patch(self.vol,64,None,self.locations[idx])
        
        HRT=Carve(low,high,self.vol)
        shape=[1]+list(HRT.shape)
        
        HRA=Carve(low,high,self.anatomy).reshape(shape)
        
        low, high, loc1 =Patch(self.vol,64*2,None,self.locations[idx])
        assert np.all(loc==loc1)
        LRT=Carve(low,high,self.vol)
        LRA=Carve(low,high,self.anatomy)
        
        
        zf=np.array(HRT.shape)/np.array(LRT.shape)
        
        LRT=zoom(LRT,zf,order=3).reshape(shape)
        LRA=zoom(LRA,zf,order=3).reshape(shape)
        
        HRT=HRT.reshape(shape)
        
        HR=np.concatenate([HRA,HRT],axis=0)
        LR=np.concatenate([LRA,LRT],axis=0)
        
        
        sample={'HD':np.copy(HR),
                'LD':np.copy(LR),
                'loc':self.locations[idx]}
        
        if type(self.GetLabels)==str:
            sample['L']=Carve(low,high,self.GT)
        
        if self.transforms:
            sample=self.transforms(sample)
        
        return sample


def Rebuild(RefVol,patches,locations,size=64):
    new=np.zeros_like(RefVol)
    scount=np.zeros_like(RefVol)
    for patch, loc in tqdm.tqdm(zip(patches,locations),desc='Building output...',total=len(patches)):
        low, high, _ = Patch(RefVol,size,None,loc)
        new[low[0]:high[0],low[1]:high[1],low[2]:high[2]] += patch
        scount[low[0]:high[0],low[1]:high[1],low[2]:high[2]] +=1
    scount[scount==0]=1
    return new/scount

def Carve(low,high,vol):
    
    return np.copy(vol[low[0]:high[0],low[1]:high[1],low[2]:high[2]])

def Patch(connected_components,size,positive,loc=None):
    if loc is None:
        t1=False
        t2=False
        while not np.any(t1*t2):
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
            
            t1=loc>(0,0,0)
            t2=loc<connected_components.size
    
    
    loc=np.round(loc)

    max_upper_indexes=np.array(connected_components.shape)-size
    
    lows=np.clip(loc-np.array(size/2).astype(int),0,max_upper_indexes).astype(int)
    
    highs=lows+size
    
    return lows, highs, loc

def GetPosFrac(dataroot):
    samples=listdata(dataroot).values()
    pos=0
    for sample in samples:
        LAB=nib.load(sample['label']).get_fdata()
        if np.sum(LAB)>0:
            pos+=1
    return pos/len(samples)

def NewPatches(dataroot,outfolder,size=64,ActuallySave=True,volpriority='_res.nii.gz',maxp=np.inf):
    si=str(size)
    samples=listdata(dataroot)
    assert type(size)==int or len(size)==3
    size=np.array((1,1,1))*np.array(size)
    datalist=[]
    outlist=os.path.join(outfolder,'databox'+str(size)+'.p')
    if os.path.isfile(outlist): datalist=pickle.load(open(outlist,'rb'))
    
    for sample in samples.values():
        
        
        if os.path.isfile(sample['struct']+volpriority):
            MRI=nib.load(sample['struct']+volpriority).get_fdata()
            TOF=nib.load(sample['TOF']+volpriority)
            LAB=nib.load(sample['label']+volpriority).get_fdata()
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
        
        locations = potential_aneurysm(TOF)
        
        if not os.path.isdir(os.path.join(outfolder, sample['ID'])): os.mkdir(os.path.join(outfolder, sample['ID']))
     
        
        for n, loc in enumerate(locations):
            if n<maxp:
            
                if os.path.isfile(os.path.join(outfolder, sample['ID'],'_STR_'+str(n)+'_'+si+'.nii.gz')): continue
                S={'ID':sample['ID'],
                   'struct':os.path.join( sample['ID'],'STR_'+str(n)+'_'+si+'.nii.gz'),
                   'TOF':os.path.join( sample['ID'],'TOF_'+str(n)+'_'+si+'.nii.gz'),
                   'labels':os.path.join( sample['ID'],'ane_'+str(n)+'_'+si+'.nii.gz'),
                   
                   'structB':os.path.join( sample['ID'],'STR_'+str(n)+'_'+si+'_big.nii.gz'),
                   'TOFB':os.path.join( sample['ID'],'TOF_'+str(n)+'_'+si+'_big.nii.gz'),
                   'labelsB':os.path.join( sample['ID'],'ane_'+str(n)+'_'+si+'_big.nii.gz')}
                    
            
                low, high, _ = Patch(connected_components,size,None,loc)
                lab=Carve(low,high,LAB)
                mri=Carve(low,high,MRI)
                tof=Carve(low,high,TOF)
                
                low, high, _ = Patch(connected_components,size*2,None,loc)
                labB=Carve(low,high,LAB)
                mriB=Carve(low,high,MRI)
                tofB=Carve(low,high,TOF)
                
                
                
                zf=np.array(lab.shape)/np.array(labB.shape)
                
                labB=zoom(labB,zf,order=0)
                mriB=zoom(mriB,zf,order=3)
                tofB=zoom(tofB,zf,order=3)
                
                assert labB.shape==lab.shape
                
                for x in [lab,labB,mri,mriB,tof,tofB]:
                    test=np.all(size==x.shape)
                    
                    if not test:
                        
                        print (low, high, loc, sample)
                        return (low, high, loc, sample)
                
                
                
                S['positive']=True if len(lab[lab==1])>0 else False
                if ActuallySave:
                    nib.save(nib.Nifti1Image(lab, aff, hea),os.path.join(outfolder, S['labels']))
                    nib.save(nib.Nifti1Image(mri, aff, hea),os.path.join(outfolder, S['struct']))
                    nib.save(nib.Nifti1Image(tof, aff, hea),os.path.join(outfolder, S['TOF']))
                    
                    nib.save(nib.Nifti1Image(labB, aff, hea),os.path.join(outfolder, S['labelsB']))
                    nib.save(nib.Nifti1Image(mriB, aff, hea),os.path.join(outfolder, S['structB']))
                    nib.save(nib.Nifti1Image(tofB, aff, hea),os.path.join(outfolder, S['TOFB']))
                    datalist.append(S)
                    
                    
                datalist.append(S)
    if ActuallySave: pickle.dump(datalist,open(outlist,'wb'))

def ExtractPatches(dataroot,outfolder,number,size=64,ActuallySave=True,volpriority='_res.nii.gz'):
    si=str(size)
    samples=listdata(dataroot)
    assert type(size)==int or len(size)==3
    size=np.array((1,1,1))*np.array(size)
    datalist=[]
    outlist=os.path.join(outfolder,'databox'+str(size)+'.p')
    if os.path.isfile(outlist): datalist=pickle.load(open(outlist,'rb'))
    
    
    pos_frac=0.8230088495575221
    
    for sample in samples.values():
        
        
        if os.path.isfile(sample['struct']+volpriority):
            MRI=nib.load(sample['struct']+volpriority).get_fdata()
            TOF=nib.load(sample['TOF']+volpriority)
            LAB=nib.load(sample['label']+volpriority).get_fdata()
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
                   'type':pos}
                    
            
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
                
                for x in [lab,labB,mri,mriB,tof,tofB]:
                    test=np.all(size==x.shape)
                    
                    if not test:
                        
                        print (low, high, ref, pos, sample)
                        return (low, high, ref, pos, sample)
                if ActuallySave:
                    nib.save(nib.Nifti1Image(lab, aff, hea),os.path.join(outfolder, S['labels']))
                    nib.save(nib.Nifti1Image(mri, aff, hea),os.path.join(outfolder, S['struct']))
                    nib.save(nib.Nifti1Image(tof, aff, hea),os.path.join(outfolder, S['TOF']))
                    
                    nib.save(nib.Nifti1Image(labB, aff, hea),os.path.join(outfolder, S['labelsB']))
                    nib.save(nib.Nifti1Image(mriB, aff, hea),os.path.join(outfolder, S['structB']))
                    nib.save(nib.Nifti1Image(tofB, aff, hea),os.path.join(outfolder, S['TOFB']))
                    datalist.append(S)
                
                
            datalist.append(S)
    if ActuallySave: pickle.dump(datalist,open(outlist,'wb'))
    
def assign(LAB,value):
    a=np.zeros_like(LAB)
    a[LAB==value]=1
    return a

def AddPatch(sampledict,root,Categories=3):
    #print(sampledict)
    #print(root)
    out={}
    LAB=nib.load(os.path.join(root,sampledict['labels'])).get_fdata()
    shape=[1]+list(LAB.shape)
    #print(shape)
    #print(LAB.shape)
    
    labels=np.zeros([Categories]+list(LAB.shape))
    #print(labels.shape)
    
    
    # labels[0,:,:,:][LAB==0]=1
    # labels[1,:,:,:][LAB==1]=1
    # labels[2,:,:,:][LAB==2]=1
    
    # for k in range(Categories):
    
    if Categories==3:
        labels[0,:,:,:]=assign(LAB,0)
        labels[1,:,:,:]=assign(LAB,1)
        labels[2,:,:,:]=assign(LAB,2)
    else:
        bop=assign(LAB,1)
        labels[1,:,:,:]=bop
        labels[0,:,:,:]=1-bop
    
    
    MRI_HD=nib.load(os.path.join(root,sampledict['struct'])).get_fdata().reshape(shape)
    MRI_LD=nib.load(os.path.join(root,sampledict['structB'])).get_fdata().reshape(shape)
    
    TOF_HD=nib.load(os.path.join(root,sampledict['TOF'])).get_fdata().reshape(shape)
    TOF_LD=nib.load(os.path.join(root,sampledict['TOFB'])).get_fdata().reshape(shape)
    
    out['HD']=np.concatenate([MRI_HD,TOF_HD],axis=0)
    out['LD']=np.concatenate([MRI_LD,TOF_LD],axis=0)
    
    out['labels']=labels
    
    
    return out


def SizeStats(datapath):
    dataset=listdata(datapath)
    dims=[]
    pixdims=[]
    for paths in tqdm.tqdm(dataset.values()):
        
        TOF=nib.load(paths['TOF'])
        H=TOF.header
        dims.append(H['dim'])
        pixdims.append(H['pixdim'])
        
    return dims, pixdims
    
def TheAllResizer(datapath,xysize=0.35714287,zsize=0.5):
    """
    560 scelto per pixdim: 1024*0.1953125/0.35714287

    """
    dataset=listdata(datapath)
    defsize=np.array((xysize,xysize,zsize))
    with tqdm.tqdm(total=len(dataset.values())) as t:
        for paths in dataset.values():
                
            MRI=nib.load(paths['struct'])
            TOF=nib.load(paths['TOF'])
            LAB=nib.load(paths['label'])
            nibs=[MRI,TOF,LAB]
            ps=[paths['struct'],paths['TOF'],paths['label']]
            
            ords=[3,3,0]
            
            for ni, o, p in zip(nibs,ords,ps):
                
                t.set_description(p)
                H=ni.header
                size=np.array((H['pixdim'][1],H['pixdim'][2],H['pixdim'][3]))
                shapefactor=size/defsize
                newvol=zoom(ni.get_fdata(),shapefactor,order=o)
                H['dim'][1]=newvol.shape[0]
                H['dim'][2]=newvol.shape[1]
                H['dim'][3]=newvol.shape[2]
                H['pixdim'][1]=xysize
                H['pixdim'][2]=xysize
                H['pixdim'][3]=zsize
                
                newnii=nib.Nifti1Image(newvol, ni.affine,header=H)
                nib.save(newnii, p+'_standard.nii.gz')
            t.update()

def InResizer(TOFp,MRIp,xysize=0.35714287,zsize=0.5):
    """
    560 scelto per pixdim: 1024*0.1953125/0.35714287

    """
    defsize=np.array((xysize,xysize,zsize))
    
                
    MRI=nib.load(MRIp)
    TOF=nib.load(TOFp)
    nibs=[MRI,TOF]
    outs=[]
    
    for ni in nibs:
        
        H=ni.header
        OrigSize=np.array((H['pixdim'][1],H['pixdim'][2],H['pixdim'][3]))
        shapefactor=OrigSize/defsize
        newvol=zoom(ni.get_fdata(),shapefactor,order=3)
        H['dim'][1]=newvol.shape[0]
        H['dim'][2]=newvol.shape[1]
        H['dim'][3]=newvol.shape[2]
        H['pixdim'][1]=xysize
        H['pixdim'][2]=xysize
        H['pixdim'][3]=zsize
        
        newnii=nib.Nifti1Image(newvol, ni.affine,header=H)
        outs.append(newnii)
    return outs,OrigSize

def OutResizer(vol,OrigSize,ResizedTOF,order=0):
    OrigSize
    H=ResizedTOF.header
    ResizedSize=np.array((H['pixdim'][1],H['pixdim'][2],H['pixdim'][3])) 
    shapefactor=ResizedSize/OrigSize
    newvol=zoom(vol,shapefactor,order=order)
    H['dim'][1]=newvol.shape[0]
    H['dim'][2]=newvol.shape[1]
    H['dim'][3]=newvol.shape[2]
    H['pixdim'][1]=OrigSize[0]
    H['pixdim'][2]=OrigSize[1]
    H['pixdim'][3]=OrigSize[2]
    newnii=nib.Nifti1Image(newvol, ResizedTOF.affine,header=H)
    return newnii

def YourFriendlyResizer(datapath,standardsize=560,standardsize2=128):
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
        newnib=[]
        ords=[3,3,0]
        if MRI.shape[0]>=600:
            print('Downsize',paths['TOF'])
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
                newnib.append(newnii)
            nibs = newnib
        
        if MRI.shape[2]<80 and MRI.header['pixdim'][3]>0.85:
            print('Upsize',paths['TOF'])
            for ni, o, p in zip(nibs,ords,ps):
                shapefactor=[1,1,standardsize2/MRI.shape[2]]
                newvol=zoom(ni.get_fdata(),shapefactor,order=o)
                print(newvol.shape)
                H=ni.header
                H['dim'][3]=standardsize2
                H['pixdim'][3]=H['pixdim'][3]*standardsize/MRI.shape[2]
                newnii=nib.Nifti1Image(newvol, ni.affine,header=ni.header)
                nib.save(newnii, p+'_res.nii.gz')
    

def listdata(datafolder):
    dataset={}
    
    for sample in os.scandir(datafolder):
        AddSampleFilePaths(sample.path,dataset)
        
    return dataset

class PatchesDataset(Dataset):
    def __init__(self,patchesroot,databoxfile,transforms=None, Testsbj=[],
                 Test=False, Categories=3,NegRatio=0.7):
        dataset=pickle.load(open(os.path.join(patchesroot,databoxfile),'rb'))
        self.NegRatio=NegRatio
        if Test:
            self.dataset=[k for k in dataset if k['ID'] in Testsbj]
        else:
            self.dataset=[k for k in dataset if k['ID'] not in Testsbj]
        self.path=patchesroot
        self.transforms=transforms
        self.Categories=Categories
        if 'positive' in self.dataset[0]:
            self.NewMode=True
            self.PosIDX=[]
            self.NegIDX=[]
            for idx, k in enumerate(self.dataset):
                if k['positive']:
                    self.PosIDX.append(idx)
                else:
                    self.NegIDX.append(idx)
                    
            
        else:
            self.NewMode=False
                
    def scamblenegs(self):
        if self.NewMode:
            S=int(len(self.PosIDX)*self.NegRatio)
            negs=np.random.choice(self.NegIDX,size=S,replace=False)
            negs=list(negs)
            self.indexes=self.PosIDX+negs
            
                
    def __len__(self):
        if self.NewMode: return len(self.indexes)
        
        return len(self.dataset)
    
    def __getitem__(self,I):
        #print(self.dataset[idx])
        #print(idx)
        #print(self.path)
        if self.NewMode: 
            idx = self.indexes(I)
        else:
            idx = I
            
        sample=AddPatch(self.dataset[idx],self.path,self.Categories)
        
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

class Flip():
    def __init__(self,prob=0.5,axis=0,order={'sample':3,'labels':0}):
        self.prob=prob
        self.ax=axis
        self.order=order
    def __call__(self,sample):
        if np.random.random()<self.prob:
            for key in self.order:
                sample[key]=np.flip(sample[key],axis=self.ax)
        return sample

class ToTensor():
    def __init__(self,device='cuda',order={'sample':3,'labels':0}):
        self.device=device
        self.order=order
    def __call__(self,sample):
        # print('ToTensor')
        # if 'loc' in sample:
        #     sample['loc']=
        for key in self.order:
            sample[key]=torch.from_numpy(sample[key]).float()#.to(self.device)
            
        
        return sample
    
        
        
