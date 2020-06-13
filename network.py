#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import numpy as np

EPS=1e-10 # log offset to avoid log(0)
class InterpWrapper(nn.Module):
    def __init__(self,in_channels,out_channels,PAR):
        super(InterpWrapper,self).__init__()
        
        self.mode=PAR['InterpMode']
    
    def forward(self,input,output_size):
        return torch.nn.functional.interpolate(input,size=output_size,mode=self.mode)


class TransposeWrapper(nn.Module):
    def __init__(self,in_channels,out_channels,PAR):
        super(TransposeWrapper,self).__init__()
        
        self.interp=nn.ConvTranspose3d(in_channels, out_channels, kernel_size=int(PAR['TransposeSize']),stride=int(PAR['TransposeStride']),padding=FindPad(PAR['TransposeSize']))
    
    def forward(self,input,output_size):
        return self.interp(input,output_size=output_size[-3:])

# def checksize(insize,kernel,stride,padding,outpad):
#     return (insize-1)*stride-2*padding+(kernel-1)+outpad+1

class PoolWrapper(nn.Module):
    def __init__(self,channels,PAR):
        super(PoolWrapper,self).__init__()
        
        self.pool=nn.MaxPool3d(PAR['PoolShape'])
    def forward(self,input):
        return self.pool(input)

class StrideConvWrapper(nn.Module):
    def __init__(self,channels,PAR):
        super(StrideConvWrapper,self).__init__()
        
        self.pool=nn.Conv3d(channels, channels, PAR['DownConvKernel'],stride=PAR['PoolShape'])
    def forward(self,input):
        return self.pool(input)

DEF_PARAMS={'FilterSize':3,
            'FiltersNumHighRes':np.array([64, 64, 64]),
            'FiltersNumLowRes':np.array([64, 64, 64]),
            'FiltersDecoder':np.array([64, 64, 64]),
            'Categories':int(3), 
            'Activation':nn.LeakyReLU, 
            'InblockSkip':False,
            'ResidualConnections':False,
            'PoolShape':2,
            'BNorm':nn.BatchNorm3d,
            'Conv':nn.Conv3d,
            'Downsample':PoolWrapper,
            'Upsample':InterpWrapper,
            'InterpMode':'trilinear',
            'DownConvKernel':3,
            'WDecay':0,
            'TransposeSize':4,
            'TransposeStride':2,
            }

DEF_yolo={'FilterSize':3,
          'FiltersNumHighRes':np.array([8, 16, 32]),
          'FiltersNumLowRes':np.array([16, 32, 64]),
          'FiltersDecoder':np.array([16, 32, 64]),
          'Categories':int(3), 
          'Activation':nn.LeakyReLU, 
          'InblockSkip':False,
          'ResidualConnections':False,
          'PoolShape':2,
          'BNorm':nn.BatchNorm3d,
          'Conv':nn.Conv3d,
          'Downsample':PoolWrapper,
          'Upsample':TransposeWrapper,
          'InterpMode':'trilinear',
          'DownConvKernel':3,
          'WDecay':0.00,
          'TransposeSize':4,
          'TransposeStride':2,
          'PositiveWeight':100,
          'CoordsWeight':1
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


    def __init__(self,FilterIn,FilterNum,PAR,FilterSize=None,Inplace=False):
        super(OneConv,self).__init__()
        
        if FilterSize== None:
            FilterSize=PAR['FilterSize']
        self.activate=PAR['Activation'](inplace=Inplace)
        self.norm=PAR['BNorm'](int(FilterNum), eps=1e-05, momentum=0.1, affine=True)
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

    def __init__(self,FilterIn,FilterNum,PAR,Inplace=False):
        super(SkipConvBlock,self).__init__()
        self.ResConn=PAR['ResidualConnections']
        self.conv1=OneConv(int(FilterIn),int(FilterNum),PAR=PAR,Inplace=Inplace)
        self.conv2=OneConv(int(FilterIn+FilterNum),int(FilterNum),PAR=PAR,Inplace=Inplace)
        self.conv3=OneConv(int(FilterIn+FilterNum*2),int(FilterNum),PAR=PAR,Inplace=Inplace)
    
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

    def __init__(self,FilterIn,FilterNum,PAR,Inplace=False):
        super(NoSkipConvBlock,self).__init__()
        self.ResConn=PAR['ResidualConnections']
        self.conv1=OneConv(int(FilterIn),int(FilterNum),PAR=PAR,Inplace=Inplace)
        self.conv2=OneConv(int(FilterNum),int(FilterNum),PAR=PAR,Inplace=Inplace)
        self.conv3=OneConv(int(FilterNum),int(FilterNum),PAR=PAR,Inplace=Inplace)
        
        if self.ResConn:
            self.outf=lambda block, first: block + first
        else:
            self.outf=lambda block, first: block
        
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        
        second=self.conv2(first)
        BlockOut=self.conv3(second)
        
        return self.outf(BlockOut,first)
    
class CascadeUp(nn.Module):
    def __init__(self,in_channels,out_channels,PAR,Inplace=False):
        super(CascadeUp,self).__init__()
        
        self.interp=nn.ConvTranspose3d(in_channels, out_channels, kernel_size=PAR['TransposeSize'],stride=PAR['TransposeStride'],padding=FindPad(PAR['TransposeSize']))
        self.bn=PAR['BNorm'](int(out_channels), eps=1e-05, momentum=0.1, affine=True)
        self.activate=PAR['Activation'](inplace=Inplace)
    
    def forward(self,input,output_size):
        x=self.interp(input,output_size=output_size)
        x=self.activate(x)
        x=self.bn(x)
        
        return x
    
    
class SkipCascadeConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,PAR,Inplace=False,UpConv=False):
        super(SkipCascadeConvBlock,self).__init__()
        self.ResConn=PAR['ResidualConnections']
        if UpConv: 
            self.UpConv=CascadeUp(FilterIn, FilterNum, PAR,Inplace)
            FilterIn=FilterNum
        else:
            self.UpConv= lambda x,y: x
        
        self.conv1=OneConv(int(FilterIn),int(FilterNum),PAR=PAR,Inplace=Inplace)
        self.conv2=OneConv(int(FilterIn+FilterNum),int(FilterNum),PAR=PAR,Inplace=Inplace)
    
        if self.ResConn:
            self.outf=lambda block, first: block + first
        else:
            self.outf=lambda block, first: block
            
    def forward(self,BlockInput,UpShape=[0,0,0,0,0]):
        
        first=self.conv1(self.UpConv(BlockInput, UpShape[-3:]))
        fconv=torch.cat((first,BlockInput),1)
        
        second=self.conv2(fconv)
        
        return self.outf(second,first)
    
class NoSkipCascadeConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,PAR,Inplace=False,UpConv=False):
        super(NoSkipCascadeConvBlock,self).__init__()
        self.ResConn=PAR['ResidualConnections']
        if UpConv: 
            self.UpConv=CascadeUp(FilterIn, FilterNum, PAR,Inplace)
            FilterIn=FilterNum
        else:
            self.UpConv= lambda x,y: x
        
        self.conv1=OneConv(int(FilterIn),int(FilterNum),PAR=PAR,Inplace=Inplace)
        self.conv2=OneConv(int(FilterNum),int(FilterNum),PAR=PAR,Inplace=Inplace)
        
        if self.ResConn:
            self.outf=lambda block, first: block + first
        else:
            self.outf=lambda block, first: block
        
    def forward(self,BlockInput,UpShape=[0,0,0,0,0]):
        
        first=self.conv1(self.UpConv(BlockInput, UpShape[-3:]))
        
        second=self.conv2(first)
        
        return self.outf(second,first)
    
class USideBranch(nn.Module):
    
    def __init__(self,depth,PARAMS,filtersin):
        super(USideBranch,self).__init__()
        
        self.depth=depth
        self.PARAMS=PARAMS
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
        else:
            ConvBlock=NoSkipConvBlock
            
        self.layerlist=nn.ModuleDict()
        
        for i in reversed(range(1,depth+1)):
            self.layerlist['Up'+str(i)]=PARAMS['Upsample'](PARAMS['FiltersDecoder'][i],PARAMS['FiltersDecoder'][i],PAR=PARAMS)
            
            self.layerlist['Dense'+str(i)]=ConvBlock(PARAMS['FiltersDecoder'][i],
                                                     PARAMS['FiltersDecoder'][i-1],
                                                     PAR=PARAMS)
        
        
        self.layerlist['Dense'+str(0)]=OneConv(PARAMS['FiltersDecoder'][0],
                                               PARAMS['Categories'],
                                               PAR=PARAMS,
                                               Inplace=True)
    def forward(self,x,outsize):
        
        for i in reversed(range(1,self.depth+1)):
            tsize=np.array(outsize)/np.power(2,i-1)
            tsize=[outsize[0],outsize[1],int(tsize[2]),int(tsize[3]),int(tsize[4])]
            x=self.layerlist['Up'+str(i)](x,tsize)
            x=self.layerlist['Dense'+str(i)](x)
        
        return self.layerlist['Dense'+str(0)](x)
    
def Upsize(chanvol,volvol):
    
    chansize=np.array(chanvol.size())
    volsize=np.array(volvol.size())
    
    return (chansize[0],chansize[1],volsize[2],volsize[3],volsize[4])
    

        
class U_Net_Like(nn.Module):
    """
    Network definition, based on unpooling
    """
    
    def __init__(self,PARAMS=DEF_PARAMS):
        super(U_Net_Like,self).__init__()
        self.PARAMS=PARAMS
        
        assert len(PARAMS['FiltersNumHighRes'])==len(PARAMS['FiltersNumLowRes'])
        assert len(PARAMS['FiltersDecoder'])==len(PARAMS['FiltersNumLowRes'])
        
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
        else:
            ConvBlock=NoSkipConvBlock
        
        self.encoder=nn.ModuleDict()
        self.decoder=nn.ModuleDict()
        self.side=nn.ModuleDict()
        
        
        self.encoder['DenseHigh'+str(0)]=ConvBlock(2,PARAMS['FiltersNumHighRes'][0],PAR=PARAMS)
        self.encoder['DenseLow'+str(0)]=ConvBlock(2,PARAMS['FiltersNumLowRes'][0],PAR=PARAMS)
        
        # self.encoder['PoolHigh'+str(0)]=PARAMS['Downsample'](PARAMS['PoolShape'],return_indices=True)
        # self.encoder['PoolLow'+str(0)]=PARAMS['Downsample'](PARAMS['PoolShape'],return_indices=True)
        
        for i in range(1,len(PARAMS['FiltersNumLowRes'])):
            self.encoder['PoolHigh'+str(i)]=PARAMS['Downsample'](PARAMS['FiltersNumHighRes'][i-1],PAR=PARAMS)
            self.encoder['PoolLow'+str(i)]=PARAMS['Downsample'](PARAMS['FiltersNumLowRes'][i-1],PAR=PARAMS)
            
            self.encoder['DenseHigh'+str(i)]=ConvBlock(PARAMS['FiltersNumHighRes'][i-1],PARAMS['FiltersNumHighRes'][i],PAR=PARAMS)
            self.encoder['DenseLow'+str(i)]=ConvBlock(PARAMS['FiltersNumLowRes'][i-1],PARAMS['FiltersNumLowRes'][i],PAR=PARAMS)
            
            
        
        self.decoder['Dense'+str(i)]=ConvBlock(PARAMS['FiltersNumHighRes'][i]+PARAMS['FiltersNumLowRes'][i],
                                              PARAMS['FiltersDecoder'][i],
                                              PAR=PARAMS)
        
        self.side[str(i)]=USideBranch(i,PARAMS,PARAMS['FiltersDecoder'][i])
        self.decoder['Up'+str(i)]=PARAMS['Upsample'](PARAMS['FiltersDecoder'][i],PARAMS['FiltersDecoder'][i],PAR=PARAMS)
        
        
        
        for i in reversed(range(1,len(PARAMS['FiltersDecoder'])-1)):
            
            self.decoder['Dense'+str(i)]=ConvBlock(PARAMS['FiltersNumHighRes'][i]+PARAMS['FiltersNumLowRes'][i]+PARAMS['FiltersDecoder'][i+1],
                                              PARAMS['FiltersDecoder'][i],
                                              PAR=PARAMS)
            
            self.side[str(i)]=USideBranch(i,PARAMS,PARAMS['FiltersDecoder'][i])
            self.decoder['Up'+str(i)]=PARAMS['Upsample'](PARAMS['FiltersDecoder'][i],PARAMS['FiltersDecoder'][i],PAR=PARAMS)
            
            
        
        self.decoder['Dense'+str(0)]=ConvBlock(PARAMS['FiltersNumHighRes'][0]+PARAMS['FiltersNumLowRes'][0]+PARAMS['FiltersDecoder'][1],
                                              PARAMS['FiltersDecoder'][0],
                                              PAR=PARAMS)
        self.side[str(0)]=USideBranch(0,PARAMS,PARAMS['FiltersDecoder'][0])
        
        
        self.Classifier=PARAMS['Conv'](PARAMS['Categories']*len(PARAMS['FiltersDecoder']),PARAMS['Categories'],1) #classifier layer
        self.softmax=nn.Softmax(dim=1)
            
            
    def forward(self,MRI_high,MRI_low):
        
        denseA={}
        denseB={}
        side=[]
        decoder={}
        Unpool={}
        
        denseA[0] = self.encoder['DenseHigh'+str(0)](MRI_high.cuda())
        denseB[0] = self.encoder['DenseLow'+str(0)](MRI_low.cuda())
        
        
        
        for i in range(1,len(self.PARAMS['FiltersNumHighRes'])):
            
            denseA[i] = self.encoder['PoolHigh'+str(i)](denseA[i-1])
            denseB[i] = self.encoder['PoolLow'+str(i)](denseB[i-1])
            
            denseA[i] = self.encoder['DenseHigh'+str(i)](denseA[i])
            denseB[i] = self.encoder['DenseLow'+str(i)](denseB[i])
        
        
        
        cat=torch.cat([denseA[i],denseB[i]],dim=1)
        decoder[i] = self.decoder['Dense'+str(i)](cat)
        
        Unpool[i]=self.decoder['Up'+str(i)](decoder[i],Upsize(decoder[i],denseA[i-1]))
        side.append(self.side[str(i)](decoder[i],MRI_high.size()))
        
        for i in reversed(range(1,len(self.PARAMS['FiltersNumHighRes'])-1)):
            cat=torch.cat([denseA[i],denseB[i],Unpool[i+1]],dim=1)
            decoder[i] = self.decoder['Dense'+str(i)](cat)
            
            Unpool[i]=self.decoder['Up'+str(i)](decoder[i],Upsize(decoder[i],denseA[i-1]))
            side.append(self.side[str(i)](decoder[i],MRI_high.size()))
            
        cat=torch.cat([denseA[0],denseB[0],Unpool[1]],dim=1)
        decoder[0] = self.decoder['Dense'+str(0)](cat)
        side.append(self.side[str(0)](decoder[0],MRI_high.size()))
        for k in range(3):
            side[k]=self.softmax(side[k])
        
        Combine=self.softmax(self.Classifier(torch.cat(side,dim=1)))
        
        return side, Combine


class CascadedDecoder(nn.Module):
    def __init__(self,PARAMS):
        super(CascadedDecoder,self).__init__()
        
        self.PARAMS=PARAMS
        
        assert len(PARAMS['FiltersNumHighRes'])==3
        assert len(PARAMS['FiltersDecoder'])==3
        
        if PARAMS['InblockSkip']:
            ConvBlock=SkipCascadeConvBlock
        else:
            ConvBlock=NoSkipCascadeConvBlock
        
        self.encoder=nn.ModuleDict()
        self.decoder=nn.ModuleDict()
        
        self.encoder['Dense High 0']=ConvBlock(2, PARAMS['FiltersNumHighRes'][0], PAR=PARAMS)
        self.encoder['Dense Low 0']=ConvBlock(2, PARAMS['FiltersNumLowRes'][0], PAR=PARAMS)
        
        self.encoder['Pool High 1']=PARAMS['Downsample'](PARAMS['FiltersNumHighRes'][0],PAR=PARAMS)
        self.encoder['Pool Low 1']=PARAMS['Downsample'](PARAMS['FiltersNumLowRes'][0],PAR=PARAMS)
        
        self.encoder['Dense High 1']=ConvBlock(PARAMS['FiltersNumHighRes'][0], PARAMS['FiltersNumHighRes'][1], PAR=PARAMS)
        self.encoder['Dense Low 1']=ConvBlock(PARAMS['FiltersNumLowRes'][0], PARAMS['FiltersNumLowRes'][1], PAR=PARAMS)
        
        self.encoder['Pool High 2']=PARAMS['Downsample'](PARAMS['FiltersNumHighRes'][1],PAR=PARAMS)
        self.encoder['Pool Low 2']=PARAMS['Downsample'](PARAMS['FiltersNumLowRes'][1],PAR=PARAMS)
        
        self.encoder['Dense High 2']=ConvBlock(PARAMS['FiltersNumHighRes'][1], PARAMS['FiltersNumHighRes'][2], PAR=PARAMS)
        self.encoder['Dense Low 2']=ConvBlock(PARAMS['FiltersNumLowRes'][1], PARAMS['FiltersNumLowRes'][2], PAR=PARAMS)
        
        
        self.decoder['Up 2 1']=ConvBlock(PARAMS['FiltersNumHighRes'][2]+PARAMS['FiltersNumLowRes'][2],
                                          PARAMS['FiltersDecoder'][0],
                                          PAR=PARAMS,
                                          UpConv=True)
        
        self.decoder['Up 1']=ConvBlock(PARAMS['FiltersNumHighRes'][1] + PARAMS['FiltersNumLowRes'][1] + PARAMS['FiltersDecoder'][0],
                                          PARAMS['FiltersDecoder'][2],
                                          PAR=PARAMS,
                                          UpConv=True)
        
        self.decoder['Up 2 2']=ConvBlock(PARAMS['FiltersDecoder'][0],
                                          PARAMS['FiltersDecoder'][1],
                                          PAR=PARAMS,
                                          UpConv=True)
        
        self.decoder['Intermediate 2']=PARAMS['Conv'](int(PARAMS['FiltersDecoder'][1]),int(PARAMS['Categories']),PARAMS['FilterSize'],padding=FindPad(PARAMS['FilterSize']) )
        self.decoder['Intermediate 1']=PARAMS['Conv'](int(PARAMS['FiltersDecoder'][2]),int(PARAMS['Categories']),PARAMS['FilterSize'],padding=FindPad(PARAMS['FilterSize']) )
        self.decoder['Intermediate 0']=PARAMS['Conv'](int(PARAMS['FiltersDecoder'][2]+PARAMS['FiltersNumHighRes'][0]+PARAMS['FiltersNumLowRes'][0]),int(PARAMS['Categories']),PARAMS['FilterSize'],padding=FindPad(PARAMS['FilterSize']))
        
        self.Classifier=PARAMS['Conv'](PARAMS['Categories']*3,PARAMS['Categories'],1)
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self,MRI_high,MRI_low):
        
        E0_h=self.encoder['Dense High 0'](MRI_high.cuda())
        E0_l=self.encoder['Dense Low 0'](MRI_low.cuda())
        
        E0=torch.cat([E0_h,E0_l],dim=1)
        
        E1_h=self.encoder['Dense High 1'](self.encoder['Pool High 1'](E0_h))
        E1_l=self.encoder['Dense Low 1'](self.encoder['Pool Low 1'](E0_l))
        
        E1=torch.cat([E1_h,E1_l],dim=1)
        
        E2_h=self.encoder['Dense High 2'](self.encoder['Pool High 2'](E1_h))
        E2_l=self.encoder['Dense Low 2'](self.encoder['Pool Low 2'](E1_l))
        
        E2=torch.cat([E2_h,E2_l],dim=1)
        
        D2_a=self.decoder['Up 2 1'](E2,E1.shape)
        D2_b=self.decoder['Up 2 2'](D2_a,E0.shape)
        
        D1 =self.decoder['Up 1'](torch.cat([D2_a,E1],dim=1),E0.shape)
        
        side=[self.decoder['Intermediate 0'](torch.cat([E0,D1],dim=1)),
              self.decoder['Intermediate 1'](D1),
              self.decoder['Intermediate 2'](D2_b)]
        
        for k in range(3):
            side[k]=self.softmax(side[k])
        
        Combine=self.softmax(self.Classifier(torch.cat(side,dim=1)))
        
        return side, Combine
    


class YoloRP(nn.Module):
    def __init__(self,PARAMS=DEF_yolo):
        super(YoloRP,self).__init__()
        
        self.mods=nn.ModuleList()
        
        self.mods.append(OneConv(2,PARAMS['FiltersNumHighRes'][0],PAR=PARAMS,Inplace=True))
        self.mods.append(PARAMS['Downsample'](PARAMS['FiltersNumHighRes'][0],PAR=PARAMS))
        
        for i in range(1,len(PARAMS['FiltersNumHighRes'])):
            self.mods.append(NoSkipConvBlock(PARAMS['FiltersNumHighRes'][i-1],PARAMS['FiltersNumHighRes'][i],PAR=PARAMS,Inplace=True))
            self.mods.append(PARAMS['Downsample'](PARAMS['FiltersNumHighRes'][i],PAR=PARAMS))
        
        
        self.objectness=PARAMS['Conv'](PARAMS['FiltersNumHighRes'][i],1,1)
        self.coords=PARAMS['Conv'](PARAMS['FiltersNumHighRes'][i],3,1)
        
        self.sigmoid1=nn.Sigmoid()
        self.sigmoid2=nn.Sigmoid()
        
    def forward(self,x):
        x=x.cuda()
        for mod in self.mods:
            x=mod(x)
            
        return self.sigmoid1(self.objectness(x)), self.sigmoid2(self.coords(x))

class TruthPooler(nn.Module):
    def __init__(self,PARAMS=DEF_yolo):
        super(TruthPooler,self).__init__()
        
        self.mods=nn.ModuleList()
        
        for k in range(len(PARAMS['FiltersNumHighRes'])):
            self.mods.append(nn.MaxPool3d(PARAMS['PoolShape']))
        
    def forward(self,x):
        for mod in self.mods:
            x=mod(x)
        
        return x

class CoordPooler(nn.Module):
    def __init__(self,PARAMS=DEF_yolo):
        super(CoordPooler,self).__init__()
        
        self.mods=nn.ModuleList()
        
        for k in range(len(PARAMS['FiltersNumHighRes'])):
            self.mods.append(nn.AvgPool3d(PARAMS['PoolShape']))
        
    def forward(self,x):
        for mod in self.mods:
            x=mod(x)
        
        return x
    
class YoLoss(nn.Module):
    def __init__(self,PARAMS):
        super(YoLoss,self).__init__()
        
        self.TP=TruthPooler(PARAMS)
        self.CP=CoordPooler(PARAMS)
        
        self.weight=PARAMS['PositiveWeight']
        self.cweight=PARAMS['CoordsWeight']
        
    def forward(self,objectness,coords,labels,truecoords):
        objectness=objectness.cuda()
        coords=coords.cuda()
        labels=labels.cuda()
        truecoords=truecoords.cuda()
        
        
        GT=self.TP(labels)
        truecoords=self.TP(truecoords)
        
        
        mse=(coords-truecoords)*GT
        mse=torch.mean(mse*mse)*self.cweight
        
        
        x=objectness
        # print(GT.shape,'GT')
        # print(x.shape,'x')
        # print(EPS)
        
        
        W=GT*self.weight
        loss_object= GT*torch.log(x+EPS)+(1-GT)*torch.log(1-x+EPS)
        loss_object*= W
        loss_object=torch.mean(loss_object)
        
        # loss_object=torch.mean((1-GT)*x) + torch.mean((GT-x)*GT)*self.weight
        
        
        
        loss=torch.mean(loss_object) + mse
        return loss
        
        
        
        


# TheBigVol=np.random.random((1,2,560,560,140))

# p=YoloRP(DEF_PARAMS).cuda()

# j=p(torch.from_numpy(TheBigVol).float().cuda())

# from skimage.transform import resize
# import tqdm

# TheBigVol=np.random.random((1,2,560,560,140))
# a=CascadedDecoder(PARAMS=DEF_PARAMS).cuda()
# wind=64
# step=5
# tot=int((560-64)/step)**2*int((140-64)/step)

# with tqdm.tqdm(total=tot) as pbar:
#     for x in range(0,560-wind,step):
#         for y in range(0,560-wind,step):
#             for z in range(0,140-wind,step):
                
#                 smol=TheBigVol[:,:,x:x+64,y:y+64,z:z+64]
#                 x=int(np.floor(x/2))
#                 y=int(np.floor(y/2))
#                 z=int(np.floor(z/2))
#                 big=TheBigVol[:,:,x:x+128,y:y+128,z:z+128]
#                 big=resize(big,(1,2,64,64,64))
#                 smol=torch.from_numpy(smol).float().cuda()
#                 big= torch.from_numpy(big).float().cuda()
#                 b=a(smol,big)
#                 pbar.update(1)
            


# in0=torch.rand(2,2,64,64,64).cuda()
# in1=torch.rand(2,2,64,64,64).cuda()
# a=U_Net_Like(PARAMS=DEF_PARAMS).cuda()
# b=a(in0,in1)


