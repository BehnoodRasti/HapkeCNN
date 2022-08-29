#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:58:49 2022


@author: Behnood
"""

import matplotlib.pyplot as plt
import os

import numpy as np
import torch
import torch.optim
import torch.nn as nn

from common import *
from UnmixArch import UnmixArch
from UtilityMine import *
import scipy.io
import scipy.linalg
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
PLOT = True
save_result=False

#%%
fname2  = "Git_Data/RayTracing/Y_clean.mat"
mat2 = scipy.io.loadmat(fname2)
img_np_gt = mat2["Y_clean"]
img_np_gt = img_np_gt.transpose(2,0,1)
[p1, nr1, nc1] = img_np_gt.shape
data = scipy.io.loadmat('Git_Data/Noisy/Ray_tracing_reflectance.mat')     
YY =data['YY']
A_true_np = data['A_true'] # true abundance
A_true_np = A_true_np.transpose(2,0,1)
E = data['E'] # true endmembers
m=1
m0=1
W=np.power((m0+m)**2*np.power(E,2)+np.multiply(1+4*m*m0*E, 1-E),0.5)-(m0+m)*E
W1=np.divide(W,1+4*m*m0*E)
EE=1-np.power(W1,2)
# Number of endmemebrs 
rmax=3#E_np.shape[1]  

tol1=1 # comment out the line below for all SNRs
#tol1=YY.shape[1]

tol2=1 
# comment out the line below for 10 runs
# tol2=YY.shape[0]

# Selecting tuning parameters for the loss function
lamb=0.1
alpha=0.0001
from tqdm import tqdm

for fi in tqdm(range(tol1)):
    for fj in tqdm(range(tol2)):
            #%%
        img_noisy_np = YY[fj][fi]
        
        img_noisy_np = img_noisy_np.transpose(2,0,1)
        [p1, nr1, nc1] = img_noisy_np.shape
        #print(compare_snr(img_np_gt, img_noisy_np))
        img_resh=np.reshape(img_noisy_np,(p1,nr1*nc1))
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC=np.diag(SS)@U
        img_resh_DN=V[:,:rmax]@V[:,:rmax].transpose(1,0)@img_resh
        img_resh_np_clip=np.clip(img_resh_DN, 0, 1)
        II,III = Endmember_extract(img_resh_np_clip,rmax)
        E_np1=img_resh_np_clip[:,II]
        #%% Set up Simulated 
        INPUT = 'noise' # 'meshgrid'
        pad = 'reflection'
        need_bias=True
        OPT_OVER = 'net' 
        
        # 
        LR1 = 0.001
        show_every = 500
        exp_weight=0.99
        
        num_iter1 = 8000
        input_depth =  img_noisy_np.shape[0]
        class CAE_EndEst(nn.Module):
            def __init__(self):
                super(CAE_EndEst, self).__init__()
                # encoding layers
                self.conv1 = nn.Sequential(
                    UnmixArch(
                            input_depth, EE.shape[1],
                            num_channels_down = [ 256],
                            num_channels_up =   [ 256],
                            num_channels_skip =    [ 4],  
                            filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
                            upsample_mode='bilinear', # downsample_mode='avg',
                            need1x1_up=False,
                            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
                )
                self.dconv4 = nn.Sequential(
                            nn.Conv2d(rmax, p1, 1,1, padding="same",bias=False),
                        )
        
            def forward(self, x):
                x = self.conv1(x)
                x1 = self.dconv4(x)
                return x, x1

        net1 = CAE_EndEst()
        net1.cuda()
        
        # Loss
        def my_loss(target, End2, alpha,lamb, out_,out_spec):
            # Albedo Loss
            W=torch.pow((m0+m)**2*torch.pow(End2,2)+torch.mul(1+4*m*m0*End2, 1-End2),0.5)-(m0+m)*End2       
            W1=torch.div(W,1+4*m*m0*End2)
            End3=1-torch.pow(W1,2)
            HR=torch.mm(End3.view(p1,rmax),out_.view(rmax,nr1*nc1))
            Temp1=1+2*m*torch.pow((1-HR),0.5)
            Temp2=1+2*m0*torch.pow((1-HR),0.5)
            out_HR=torch.div(HR, torch.mul(Temp1,Temp2))
            loss = 0.5*torch.norm((out_HR.view(1,p1,nr1,nc1) - target), 'fro')**2
            #---- Net Loss-------
            loss1 = 0.5*torch.norm((out_spec.view(1,p1,nr1,nc1) - target), 'fro')**2
            #------Minimum Volume Penalty: TV-----
            O = torch.from_numpy(np.zeros((p1, rmax))).type(dtype)
            B = np_to_torch(np.identity(rmax) - np.ones((rmax))/rmax).type(dtype)
            loss2 = torch.norm(torch.mm(End3,B.view((rmax,rmax)))-O, 'fro')**2
            return loss+alpha*loss1+lamb*loss2
        img_noisy_torch = torch.from_numpy(img_resh_DN).view(1,p1,nr1,nc1).type(dtype)
        net_input1 = get_noise(input_depth, INPUT,
            (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()
        E11=np.random.rand(p1,rmax);
        E_torch = torch.from_numpy(E_np1).type(dtype)
        #%%
        out_avg = True
        
        i = 0
        def closure1():
            
            global i, out_LR_np, out_avg, out_avg_np, Eest
            
            out_LR,out_spec = net1(net_input1)
            # Smoothing
            if out_avg is None:
                out_avg = out_LR.detach()
            else:
                out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)

        #%%
            total_loss = my_loss(img_noisy_torch, net1.dconv4[0].weight.view(p1,rmax),alpha, lamb,out_LR,out_spec)
            total_loss.backward()
            print ('Iteration %05d    Loss %f ' % (i, total_loss.item()), '\r', end='')
            if  PLOT and i % show_every == 0:

                out_LR_np = out_LR.detach().cpu().squeeze().numpy()
                out_avg_np = out_avg.detach().cpu().squeeze().numpy()
                out_LR_np = np.clip(out_LR_np, 0, 1)
                out_avg_np = np.clip(out_avg_np, 0, 1) 
                f, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(10,10))
                ax1.imshow(np.stack((out_LR_np[2,:,:],out_LR_np[1,:,:],out_LR_np[0,:,:]),2))
                ax2.imshow(np.stack((out_avg_np[2,:,:],out_avg_np[1,:,:],out_avg_np[0,:,:]),2))
                plt.show()     
                
            i += 1       
            return total_loss
        net1.dconv4[0].weight=torch.nn.Parameter(E_torch.view(p1,rmax,1,1))       
        p11 = get_params(OPT_OVER, net1, net_input1)
        optimizer = torch.optim.Adam(p11, lr=LR1, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay= 0, amsgrad=False)
        for j in range(num_iter1):
                optimizer.zero_grad()
                closure1()  
                optimizer.step()
                net1.dconv4[0].weight.data[net1.dconv4[0].weight <= 0] = 0
                net1.dconv4[0].weight.data[net1.dconv4[0].weight >= 1] = 1
                if j>0:
                  Eest=net1.dconv4[0].weight.detach().cpu().squeeze().numpy()
                  if PLOT and j % show_every== 0: 
                     plt.plot(Eest)
                     plt.show()
                  
        out_avg_np = out_avg.detach().cpu().squeeze().numpy()
       

    #%%
        if  save_result is True:
                  scipy.io.savemat("ResultsNoPure/EestdB%01d%01d.mat" % (fi+2, fj+1),
                                    {'Eest%01d%01d' % (fi+2, fj+1):Eest})
                  scipy.io.savemat("ResultsNoPure/out_avg_npdB%01d%01d.mat" % (fi+2, fj+1),
                                    {'out_avg_np%01d%01d' % (fi+2, fj+1):out_avg_np.transpose(1,2,0)})
        #
