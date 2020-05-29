#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import numpy as np


class InterpWrapper(nn.Module):
    def __init__(self,in_channels,out_channels,PAR):
        super(InterpWrapper,self).__init__()
        
        self.mode=PAR['InterpMode']
    
    def forward(self,input,output_size):
        return torch.nn.functional.interpolate(input,size=output_size,mode=self.mode)
    
# torch.nn.functional.interpolate(BotNeck,size=dense[i].size()[2:],mode=self.PARAMS['InterpMode'])
# torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size) # output_size during forward pass

class TransposeWrapper(nn.Module):
    def __init__(self,in_channels,out_channels,PAR):
        super(TransposeWrapper,self).__init__()
        
        self.interp=nn.ConvTranspose3d(in_channels, out_channels, kernel_size=PAR['TransposeSize'],stride=PAR['TransposeStride'])
    
    def forward(self,input,output_size):
        return self.interp(input,output_size=output_size)



DEF_PARAMS={'FilterSize':3,
            'FiltersNumHighRes':np.array([16, 32, 64, 64]),
            'FiltersNumLowRes':np.array([16, 32, 64, 64]),
            'FiltersDecoder':np.array([16, 32, 64, 64]),
            'ClassFilters':int(64), 
            'Depth':int(4),
            'Activation':nn.LeakyReLU, 
            'InblockSkip':False,
            'ResidualConnections':False,
            'PoolShape':2,
            'BNorm':nn.BatchNorm3d,
            'Conv':nn.Conv3d,
            'Downsample':nn.MaxPool3d,
            'Upsample':TransposeWrapper,
            'InterpMode':'trilinear',
            'TransposeSize':4,
            'TransposeStride':2,
            }
# box da 48?

def FindPad(FilterSize):
    """
    Returns appropriate padding based on filter size
    """
    A=(np.array(FilterSize)-1)/2
    if type(FilterSize)==tuple:
        return tuple(A.astype(int))
    else:
        return int(A)

class OneConv(nn.Module):
    """
    Performs one single convolution bit: convolve, activate, normalize
    """


    def __init__(self,FilterIn,FilterNum,PAR,FilterSize=None):
        super(OneConv,self).__init__()
        
        if FilterSize== None:
            FilterSize=PAR['FilterSize']
        self.activate=PAR['Activation']()
        self.norm=PAR['BNorm'](int(FilterIn), eps=1e-05, momentum=0.1, affine=True)
        self.conv=PAR['Conv'](int(FilterIn),int(FilterNum),FilterSize,padding=FindPad(PAR['FilterSize']) )
        
    def forward(self,layer):
        
        x=self.conv(layer)
        x=self.activate(x)
        x=self.norm(x)
        
        return x


class SkipConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,PAR):
        super(SkipConvBlock,self).__init__()
        self.ResConn=PAR['ResidualConnections']
        self.conv1=OneConv(int(FilterIn),int(FilterNum),PAR=PAR)
        self.conv2=OneConv(int(FilterIn+FilterNum),int(FilterNum),PAR=PAR)
        self.conv3=OneConv(int(FilterIn+FilterNum*2),int(FilterNum),PAR=PAR)
    
        if self.ResConn:
            self.outf=lambda block, first: block + first
        else:
            self.outf=lambda block, first: block
            
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        fconv=torch.cat((first,BlockInput),1)
        
        second=self.conv2(fconv)
        sconv=torch.cat((first,second,BlockInput),1)
        BlockOut=self.conv3(sconv)
        
        return self.outf(BlockOut,first)
    
class NoSkipConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,PAR):
        super(NoSkipConvBlock,self).__init__()
        self.ResConn=PAR['ResidualConnections']
        self.conv1=OneConv(int(FilterIn),int(FilterNum),PAR=PAR)
        self.conv2=OneConv(int(FilterNum),int(FilterNum),PAR=PAR)
        self.conv3=OneConv(int(FilterNum),int(FilterNum),PAR=PAR)
        
        if self.ResConn:
            self.outf=lambda block, first: block + first
        else:
            self.outf=lambda block, first: block
        
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        
        second=self.conv2(first)
        BlockOut=self.conv3(second)
        
        return self.outf(BlockOut,first)

        
class Segmenter(nn.Module):
    """
    Network definition, based on unpooling
    """
    
    def __init__(self,PARAMS=DEF_PARAMS):
        super(Segmenter,self).__init__()
        self.PARAMS=PARAMS
        
        assert len(PARAMS['FiltersNumHighRes'])==len(PARAMS['FiltersNumLowRes'])
        assert len(PARAMS['FiltersDecoder'])==len(PARAMS['FiltersNumLowRes'])
        
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
            self.skipper=True
        else:
            ConvBlock=NoSkipConvBlock
            self.skipper=False
        
        self.encoder=nn.ModuleDict()
        self.decoder=nn.ModuleDict()
        
        
        
        self.encoder['DenseHigh'+str(0)]=ConvBlock(2,PARAMS['FiltersNumHighRes'][0],PAR=PARAMS)
        self.encoder['DenseLow'+str(0)]=ConvBlock(2,PARAMS['FiltersNumLowRes'][0],PAR=PARAMS)
        
        # self.encoder['PoolHigh'+str(0)]=PARAMS['Downsample'](PARAMS['PoolShape'],return_indices=True)
        # self.encoder['PoolLow'+str(0)]=PARAMS['Downsample'](PARAMS['PoolShape'],return_indices=True)
        
        for i in range(1,len(PARAMS['FiltersNumLowRes'])):
            self.encoder['PoolHigh'+str(i)]=PARAMS['Downsample'](PARAMS['PoolShape'])
            self.encoder['PoolLow'+str(i)]=PARAMS['Downsample'](PARAMS['PoolShape'])
            
            self.encoder['DenseHigh'+str(i)]=ConvBlock(PARAMS['FiltersNumHighRes'][i-1],PARAMS['FiltersNumHighRes'][i],PAR=PARAMS)
            self.encoder['DenseLow'+str(i)]=ConvBlock(PARAMS['FiltersNumLowRes'][i-1],PARAMS['FiltersNumLowRes'][i],PAR=PARAMS)
            
            
        
        self.decoder['Dense'+str(i)]=ConvBlock(PARAMS['FiltersNumHighRes'][i]+PARAMS['FiltersNumLowRes'][i],
                                              PARAMS['FiltersDecoder'][i],
                                              PAR=PARAMS)
        
        self.decoder['Up'+str(i)]=PARAMS['Upsample'](PARAMS['FiltersDecoder'][i],PARAMS['FiltersDecoder'][i],PAR=PARAMS)
        
        for i in reversed(range(1,len(PARAMS['FiltersDecoder'])-1)):
            
            self.decoder['Dense'+str(i)]=ConvBlock(PARAMS['FiltersNumHighRes'][i]+PARAMS['FiltersNumLowRes'][i]+PARAMS['FiltersDecoder'][i+1],
                                              PARAMS['FiltersDecoder'][i],
                                              PAR=PARAMS)
            
            self.decoder['Up'+str(i)]=PARAMS['Upsample'](PARAMS['FiltersDecoder'][i],PARAMS['FiltersDecoder'][i],PAR=PARAMS)
            
            
        
        self.decoder['Dense'+str(0)]=ConvBlock(PARAMS['FiltersNumHighRes'][0]+PARAMS['FiltersNumLowRes'][0]+PARAMS['FiltersDecoder'][1],
                                              PARAMS['FiltersDecoder'][0],
                                              PAR=PARAMS)
        ################# need to add the cascaded friends
        self.layers['Classifier']=PARAMS['Conv'](PARAMS['ClassFilters'],PARAMS['Categories'],1) #classifier layer
        self.layers['BinaryMask']=PARAMS['Conv'](PARAMS['ClassFilters'],1,1) #binary mask classifier
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
        #self.sigmoid1=nn.Sigmoid()
            
            
    def forward(self,MRI):
        pools={}
        dense={}
        dense[0] = self.layers['Dense_Down'+str(0)](MRI)
        dense[1], pools[0] = self.layers['Pool'+str(0)](dense[0])
        
        for i in range(1,self.PARAMS['Depth']):
            dense[i] = self.layers['Dense_Down'+str(i)](dense[i])
            dense[i+1], pools[i] = self.layers['Pool'+str(i)](dense[i])
        
        BotNeck = self.layers['Bneck'](dense[i+1])
        
        Updense={}
        Unpool={}
        
        Unpool[i] = self.layers['Up'+str(i)](BotNeck,pools[i],output_size=dense[i].size())
        cat=torch.cat([Unpool[i],dense[i]],dim=1)
        Updense[i] = self.layers['Dense_Up'+str(i)](cat)
        
        for i in reversed(range(self.PARAMS['Depth']-1)):
            
            Unpool[i]=self.layers['Up'+str(i)](Updense[i+1],pools[i],output_size=dense[i].size())
            cat=torch.cat([Unpool[i],dense[i]],dim=1)
            Updense[i]=self.layers['Dense_Up'+str(i)](cat)
            
        MultiClass=self.layers['Classifier'](Updense[0])
        MonoClass=self.layers['BinaryMask'](Updense[0])
        #MonoClassRef=self.layers['BinaryMask1'](Updense[0])
        
        Mask=self.sigmoid(MonoClass)
        
        
        return Mask, self.softmax(MultiClass)#, self.sigmoid1(MonoClassRef)


