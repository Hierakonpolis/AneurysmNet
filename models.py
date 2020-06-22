#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:17:03 2020

@author: cat
"""

from network import YoloRP, YoLoss, TruthPooler
from radam import RAdam
import numpy as np
import torch, tqdm, time, os

EPS=1e-10

def DiceLoss(Ytrue,Ypred):

    DICE = -torch.div( torch.sum(torch.mul(torch.mul(Ytrue,Ypred),2)),
                      torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue))+1)
    
    return DICE
def CCE(Ytrue,Ypred,CatW):
    shape=Ytrue.shape
    CCE=-torch.mul(Ytrue,torch.log(Ypred + EPS))
    W=torch.tensor(CatW).reshape((1,len(CatW),1,1,1)).expand(shape).float()
    
    W.requires_grad=False
    
    wCCE=torch.mul(W,CCE)
    
    return torch.mean(wCCE)

def Dice(labels,Ypred):
    
    labels [np.where(labels == np.amax(labels,axis=1))] = 1
    labels[labels!=1]=0
    
    dice=2*(np.sum(labels*Ypred,(0,2,3,4))+1)/(np.sum((labels+Ypred),(0,2,3,4))+1)
    
    return dice

class YOLOr():
    
    def __init__(self,savefile=None,parameters=None,groups=None,Testgroup=None,device='cuda'):
        
        if savefile and os.path.isfile(savefile):
            self.load(savefile)
        else:
            self.opt={}
            
            self.opt['TestGroup']=Testgroup
            self.opt['PAR']=parameters
            self.opt['device']=device
            self.opt['groups']=groups
            self.opt['Epoch']=0
            self.opt['TrainingLoss']=[]
            self.opt['TestLoss']=[]
            self.opt['TotalTime']=0
            self.opt['BestLoss']=np.inf
            self.opt['BestLossEpoch']=0
            
            self.network=YoloRP(self.opt['PAR']).to(device)
            self.optimizer=RAdam(self.network.parameters(),weight_decay=self.opt['PAR']['WDecay'])
            
        
        
        self.loss=YoLoss(self.opt['PAR'])
        
    
    
    def save(self,path):
        
        torch.save({'opt':self.opt,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    path)
        
    def load(self,path):
        checkpoint=torch.load(path)
        self.opt=checkpoint['opt']
        
        self.network=YoloRP(self.opt['PAR']).to(self.opt['device'])
        self.optimizer=RAdam(self.network.parameters(),weight_decay=self.opt['PAR']['WDecay'])
        
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
    def train_one_epoch(self,dataloader):
        
        self.network.train()
        
        losses=[]
        
        for sample in tqdm.tqdm(dataloader,total=len(dataloader),desc='Training...'):
            
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            
            # objectness= self.network(sample['sample'])
            objectness,coords = self.network(sample['sample'])
            
            # loss = self.loss(objectness,sample['labels'])
            loss = self.loss(objectness,coords,sample['labels'],sample['coords'])
            
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        
        self.opt['Epoch']+=1
        self.opt['TrainingLoss'].append(np.mean(losses))
        print(flush=True)
        return np.mean(losses)
    
    def test(self,dataloader):
        losses=[]
        
        self.network.eval()
        
        with torch.no_grad():
            for sample in tqdm.tqdm(dataloader,total=len(dataloader),desc='Testing...'):
                torch.cuda.empty_cache()
                
                objectness,coords = self.network(sample['sample'])
                # objectness= self.network(sample['sample'])
                # loss = self.loss(objectness,sample['labels'])
                loss = self.loss(objectness,coords,sample['labels'],sample['coords'])
                losses.append(float(loss.detach().cpu()))
        
        self.opt['TestLoss'].append((self.opt['Epoch'],np.mean(losses)))
        print(flush=True)

        return np.mean(losses)
    
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
            print('Training set running mean loss: ',trainloss,flush=True)
            
            testloss=self.false_pos_eval(test_dataloader)
            print('Test set loss: ',testloss,flush=True)
            self.opt['TotalTime'] += time.time()-start
            
            if testloss < self.opt['BestLoss']: 
                self.opt['BestLoss']= testloss
                self.opt['BestLossEpoch']=self.opt['Epoch']
                self.save(savebest)
            
            self.save(saveprogress)
            
    
    
    def false_pos_eval(self,dataloader,thresholds=[0.25,0.5,0.75,0.9]):#np.arange(0.1,1,0.1)):
        
        sensitivity=np.array([])
        vals=np.array([])
        crunch=TruthPooler(self.opt['PAR'])
        true=[]
        losses=[]
        
        self.network.eval()
        
        with torch.no_grad():
            for sample in tqdm.tqdm(dataloader,total=len(dataloader),desc='Testing...'):
                torch.cuda.empty_cache()
                
                # objectness = self.network(sample['sample'])
                objectness,coords = self.network(sample['sample'])
                # loss = self.loss(objectness,sample['labels'])
                loss = self.loss(objectness,coords,sample['labels'],sample['coords'])
                losses.append(float(loss.detach().cpu()))
                objectness=objectness.detach().cpu().numpy().ravel()
                labels=crunch(sample['labels']).detach().cpu().numpy().ravel()
                sensi= objectness[labels==1]
                sensitivity=np.concatenate([sensitivity,sensi])
                true=np.concatenate([true,labels])
                
                
                
                vals=np.concatenate([vals,objectness])
            print(flush=True)
            
        self.opt['TestLoss'].append((self.opt['Epoch'],np.mean(losses)))
        for T in thresholds:
            TP=np.sum(sensitivity>T)
            FP=np.sum(vals[true==0]>T)
            FPr=FP/len(true)
            S=TP/len(sensitivity)
            # ALLPOS=np.sum(vals>T)
            
            print('\nFor threshold',T,
                  # '\nTrue Positives:',TP,
                  # '\nFalse Positives:',FP,
                  '\nFalse Positive rate:',FPr,
                  '\nSensitivity:',S,
                  # '\nTotal Positives:',ALLPOS,
                  flush=True)
        return np.mean(losses)
    
    def inferece(self):
        pass

class Segmentation():
    
    def __init__(self,network,savefile=None,parameters=None,trainset=None,testset=None,device='cuda'):
        
        if savefile and os.path.isfile(savefile):
            self.load(savefile,network)
        else:
            self.opt={}
            
            self.opt['PAR']=parameters
            self.opt['device']=device
            self.opt['trainset']=trainset
            self.opt['testset']=testset
            self.opt['Epoch']=0
            self.opt['TrainingLoss']=[]
            self.opt['TestDices']=[]
            self.opt['TestLoss']=[]
            self.opt['TotalTime']=0
            self.opt['BestLoss']=np.inf
            self.opt['BestLossEpoch']=0
            
            self.network=network(self.opt['PAR']).to(self.opt['device'])
            self.optimizer=RAdam(self.network.parameters(),weight_decay=self.opt['PAR']['WDecay'])
            
    
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

        return np.mean(losses)
    
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
            print('Training set running mean loss: ',trainloss,flush=True)
            
            testloss=self.test(test_dataloader)
            print('Test set loss: ',testloss,flush=True)
            self.opt['TotalTime'] += time.time()-start
            
            if testloss < self.opt['BestLoss']: 
                self.opt['BestLoss']= testloss
                self.opt['BestLossEpoch']=self.opt['Epoch']
                self.save(savebest)
            
            self.save(saveprogress)
            
    def loss(self,sidebranches,output,GT):
        
        GT=GT.to(self.opt['device'])
        loss=DiceLoss(GT,output) + self.opt['PAR']['CCEweight']*CCE(GT, output, self.opt['PAR']['Weights'])
        for x in sidebranches:
            loss+=(DiceLoss(GT,x) + self.opt['PAR']['CCEweight']*CCE(GT, x, self.opt['PAR']['Weights']))*self.opt['PAR']['SideBranchWeight']
        return loss
    
    def inferece(self,input):
        return self.network(input)
