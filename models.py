#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:17:03 2020

@author: cat
"""

from radam import RAdam
import numpy as np
import torch, tqdm, time, os
from dataset import Rebuild

EPS=1e-10

def DiceLoss(Ytrue,Ypred):

    DICE = -torch.div( torch.sum(torch.mul(torch.mul(Ytrue,Ypred),2)),
                      torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue))+1)
    
    return DICE

class GeneralizedDice():
    def __init__(self, classs=(0,1)):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = classs

    def __call__(self, probs, target):
        
        

        pc = probs[:, self.idc, ...].type(torch.cuda.FloatTensor)
        tc = target[:, self.idc, ...].type(torch.cuda.FloatTensor)

        
        w = 1 / ((torch.einsum("bcdwh->bc", tc).type(torch.cuda.FloatTensor) + 1e-10) ** 2)
        intersection = w * torch.einsum("bcdwh,bcdwh->bc", pc, tc)
        union = w * (torch.einsum("bcdwh->bc", pc) + torch.einsum("bcdwh->bc", tc))

        divided = 1 - 2 * (torch.einsum("bc->b", intersection) + 1e-10) / (torch.einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


def Dice(labels,Ypred):
    
    labels [np.where(labels == np.amax(labels,axis=1))] = 1
    labels[labels!=1]=0
    
    dice=2*(np.sum(labels*Ypred,(0,2,3,4))+1)/(np.sum((labels+Ypred),(0,2,3,4))+1)
    
    return dice


class SurfaceLoss():
    def __init__(self, classs=1):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = classs

    def __call__(self, probs, ground_truth):

        pc = probs[:, self.idc, ...].type(torch.cuda.FloatTensor)
        dc = ground_truth[:, self.idc, ...].type(torch.cuda.FloatTensor)

        multipled = torch.einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss
SL=SurfaceLoss()
GD=GeneralizedDice()
# def GD(a,b):
#     return 0

Z1=torch.tensor([[[ 1,  1, 1],
                     [ 1,  2, 1],
                     [ 1,  1, 1]],
            
                    [[ 0,  0, 0],
                     [ 0,  0, 0],
                     [ 0,  0, 0]],
            
                    [[ -1,  -1, -1],
                     [ -1,  -2, -1],
                     [ -1,  -1, -1]]],
                     requires_grad=False)#.type(torch.DoubleTensor)
    
X1=torch.tensor([[[ 1,  0, -1],
                     [ 1,  0, -1],
                     [ 1,  0, -1]],
            
                    [[ 1,  0, -1],
                     [ 2,  0, -2],
                     [ 1,  0, -1]],
            
                    [[ 1,  0, -1],
                     [ 1,  0, -1],
                     [ 1,  0, -1]]],
                     requires_grad=False)#.type(torch.DoubleTensor)
Y1=torch.tensor([[[ 1,  1, 1],
                     [ 0,  0, 0],
                     [ -1,  -1, -1]],
            
                    [[ 1,  2, 1],
                     [ 0,  0, 0],
                     [ -1,  -2, -1]],
            
                    [[ 1,  1, 1],
                     [ 0,  0, 0],
                     [ -1,  -1, -1]]],
                     requires_grad=False)#.type(torch.DoubleTensor)
    
def Sobel(Convolveme):
    
    X=X1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    Y=Y1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    
    Z=Z1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    Xconv=torch.nn.functional.conv3d(Convolveme,X,groups=Convolveme.shape[1])
    Yconv=torch.nn.functional.conv3d(Convolveme,Y,groups=Convolveme.shape[1])
    Zconv=torch.nn.functional.conv3d(Convolveme,Z,groups=Convolveme.shape[1])
    conv=torch.abs(torch.nn.functional.pad(Xconv+Yconv+Zconv,(1,1,1,1,1,1)))
    conv[conv>0]=1
    
        
    return conv

def CCE(Ytrue,Ypred,CatW,SobW=0):
    shape=Ytrue.shape
    CCE=-torch.mul(Ytrue,torch.log(Ypred + EPS))
    W=torch.tensor(CatW).reshape((1,len(CatW),1,1,1)).expand(shape).float()
    W=W*(1+Sobel(Ytrue)*SobW)
    
    W.requires_grad=False
    
    wCCE=torch.mul(W,CCE)
    
    return torch.mean(wCCE)

class Segmentation():
    
    def __init__(self,network,savefile=None,parameters=None,testset=None,device='cuda'):
        
        if savefile and os.path.isfile(savefile):
            self.load(savefile,network)
        else:
            self.opt={}
            
            self.opt['PAR']=parameters
            self.opt['device']=device
            self.opt['testset']=testset
            self.opt['Epoch']=0
            self.opt['TrainingLoss']=[]
            self.opt['TestDices']=[]
            self.opt['TestLoss']=[]
            self.opt['TotalTime']=0
            self.opt['BestLoss']=np.inf
            self.opt['BestLossEpoch']=0
            self.opt['BestDice']=0
            
            self.network=network(self.opt['PAR']).to(self.opt['device'])
            self.optimizer=RAdam(self.network.parameters(),weight_decay=self.opt['PAR']['WDecay'])
            
        self.lrscheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.5)
    
    def save(self,path):
        
        torch.save({'opt':self.opt,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    path)
        
    def load(self,path,network):
        checkpoint=torch.load(path)
        self.opt=checkpoint['opt']
        
        self.network=network(self.opt['PAR']).to(self.opt['device'])
        self.optimizer=RAdam(self.network.parameters(),weight_decay=self.opt['PAR']['WDecay'])
        
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
    def train_one_epoch(self,dataloader):
        
        self.network.train()
        
        losses=[]
        
        for sample in tqdm.tqdm(dataloader,total=len(dataloader),desc='Training...'):
            
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            
            sidebranches,combined= self.network(sample['HD'],sample['LD'])
            
            loss = self.loss(sidebranches,combined,sample['labels'])
            
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        
        self.opt['Epoch']+=1
        self.opt['TrainingLoss'].append(np.mean(losses))
        print(flush=True)
        return np.mean(losses)
    
    def test(self,dataloader):
        losses=[]
        dices=[]
        self.network.eval()
        
        with torch.no_grad():
            for sample in tqdm.tqdm(dataloader,total=len(dataloader),desc='Testing...'):
                torch.cuda.empty_cache()
                
                sidebranches,combined= self.network(sample['HD'],sample['LD'])
                
                loss = self.loss(sidebranches,combined,sample['labels'])
                losses.append(float(loss.detach().cpu()))
                dices.append(Dice(sample['labels'].detach().cpu().numpy(),combined.detach().cpu().numpy()))
        
        dices=np.array(dices).mean(axis=0)
        
        
        self.opt['TestLoss'].append((self.opt['Epoch'],np.mean(losses)))
        self.opt['TestDices'].append((self.opt['Epoch'],dices))
        print(flush=True)
        print('Test set Dices:',dices,flush=True)

        return np.mean(losses), dices[1]
    
    def train(self,
              train_dataloader,
              test_dataloader,
              max_epochs,
              patience,
              max_time,
              saveprogress,
              savebest):
        
        
        while self.opt['Epoch']<max_epochs and self.opt['TotalTime']<max_time and (self.opt['Epoch']-self.opt['BestLossEpoch'])<patience:
            start=time.time()
            print('Epoch',self.opt['Epoch'],flush=True)
            trainloss = self.train_one_epoch(train_dataloader)
            train_dataloader.dataset.scamblenegs()
            print('Training set running mean loss: ',trainloss,flush=True)
            
            testloss, dice = self.test(test_dataloader)
            test_dataloader.dataset.scamblenegs()
            print('Test set loss: ',testloss,flush=True)
            self.opt['TotalTime'] += time.time()-start
            
            if testloss < self.opt['BestLoss']: 
                self.opt['BestLoss']= testloss
                self.opt['BestLossEpoch']=self.opt['Epoch']
                self.save(savebest)
            
            if dice > self.opt['BestDice']:
                savedice = savebest.rstrip('.pth')+'_dice.pth'
                self.opt['BestDice'] = dice
                self.save(savedice)
            
            self.save(saveprogress)
            self.lrscheduler.step()
            
    def loss(self,sidebranches,output,GT):
        
        GT=GT.to(self.opt['device'])
        loss=DiceLoss(GT,output) \
            + self.opt['PAR']['CCEweight']*CCE(GT, output, self.opt['PAR']['Weights'],SobW=self.opt['PAR']['SobelWeight']) \
            + self.opt['PAR']['SurfaceLossWeight']*SL(output,GT) \
            + self.opt['PAR']['GenDiceWeight']*GD(output,GT)
        
        for x in sidebranches:
            loss+=(DiceLoss(GT,x) \
                + self.opt['PAR']['CCEweight']*CCE(GT, x, self.opt['PAR']['Weights'],SobW=self.opt['PAR']['SobelWeight']))*self.opt['PAR']['SideBranchWeight'] \
                + self.opt['PAR']['SurfaceLossWeight']*SL(output,GT)*self.opt['PAR']['SideBranchWeight']
        return loss
    
    def inferece(self,inputloader,PreThreshold=True,FinalThreshold=0.5):
        samples=[]
        locations=[]
        self.network.eval()
        
        with torch.no_grad():
            for i, sample in tqdm.tqdm(enumerate(inputloader),total=len(inputloader),desc='Inference...'):
                torch.cuda.empty_cache()
                
                sidebranches,combined= self.network(sample['HD'],sample['LD'])
                
                samples.append(combined.detach().cpu().numpy())
                locations.append(sample['loc'].detach().cpu().numpy())
        
        patches=[]
        centerpoints=[]
        
        for k in tqdm.tqdm(range(len(samples)),desc='Patches...'):
            for K in np.split(samples[k], samples[k].shape[0],axis=0):
                # print(K.shape)
                labels=K.reshape((K.shape[1],64,64,64))
                
                if type(PreThreshold)==float:
                    L=labels[1,:,:,:]
                    L[L>PreThreshold]=1
                    labels[1,:,:,:]=L
                
                if PreThreshold:
                
                    labels [np.where(labels== np.amax(labels,axis=0))] = 1
                    labels[labels!=1]=0
                
                patches.append(labels[1,:,:,:])
                
            for K in np.split(locations[k], locations[k].shape[0],axis=0):
                centerpoints.append(K.reshape(3).astype(int))
                
        # return patches, centerpoints
        res=Rebuild(inputloader.dataset.vol, patches, centerpoints)
        
        if FinalThreshold:
            res[res>FinalThreshold]=1
            res[res!= 1]=0
        
        return res
    
