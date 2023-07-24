# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:59:46 2023

@author: YFGI6212
"""
import torch
import torch.nn as nn
import snntorch as snn

import matplotlib.pyplot as plt
import math

import pandas as pd
import numpy as np

from tqdm import tqdm
#from Layers import RLeakyNoBias as RLeaky




class network(nn.Module):
    def __init__(self,num_hidden,F,D,beta_x,beta_v,thresholds=None):
        
        super().__init__()
        
        self._F = F
        self._D = D
        self._W = -self._F.matmul(self._D)
        self._beta_x = beta_x
        self._beta_v = beta_v
        self._thresholds = thresholds
        self._num_hidden = num_hidden
        
        
        self.input_layer = nn.Linear(1,num_hidden,bias=True)
        
        self.lif1 = snn.RLeaky(beta=self._beta_v,
                            V= self._thresholds,
                            linear_features=num_hidden,
                            init_hidden=True,
                            reset_mechanism = "zero",
                            output = True)
        
        self.output_layer = nn.Linear(self._num_hidden,1,bias=True)
        
        self.readout = snn.Leaky(self._beta_x,
                              init_hidden=True,
                              output = True,
                              #threshold = 1,
                              reset_mechanism = 'none'
                              )
        
        
        # initialize the linear recurrent
        
        with torch.no_grad():
            self.input_layer.weight.copy_(self._F)
            self.lif1.recurrent.weight.copy_(self._W)
            self.output_layer.weight.copy_(self._D)
        

    def forward(self,x):
        """
        Parameters
        ----------
        x : torch.tensor: Timeseries sequences
            DESCRIPTION: shape: [batch_size,time_steps,variates]
            
        Returns: spikes, membranes postentials history
        -------
        None.
        
        !! HERE WE ASSUME ONLY ONE BATCH AND DO WRITE x = [time_step,variates]
        """
        
        n_steps = x.shape[0]
        
        spk_lc1    = []
        mem_lc1    = []
        mem_output = []
        
        previous = torch.zeros(1)
        
        # Initalize membrane potential
        mem_1 = self.lif1.init_rleaky()
        mem_2 = self.readout.init_leaky()
        
        for step in range(n_steps):
            y = self.input_layer(x[step])
            spk_1, mem1 = self.lif1(y)
            spk_lc1.append(spk_1)
            mem_lc1.append(mem1)
            
            
            spk2, mem2 = self.readout(spk_1)
            y = self.output_layer(mem2)
            
            mem_output.append(y)
        return(mem_output,spk_lc1,mem_lc1)
        
        
### First net

np.random.seed(3425)
torch.manual_seed(124)

N_steps = 10000
Data = pd.read_csv("data/VanDerPol-1.csv")


beta_x = 0.8
beta_v = 0.8

mu = 0.01
nu = 0.02

x = Data['z'].to_numpy()


N = 200 # neurons population

F = torch.tensor(np.random.normal(0,1.2,(N,1)))/math.sqrt(N)
#F = torch.tensor(np.random.uniform(-1.2,1.2,(N,1)))/math.sqrt(N)

W = -F.matmul(F.t())

Thresholds = np.array([1/2*(F[i,0]*F[i,0] + mu + nu) for i in range(N)])



net = network(N,F,F.t(),beta_x,beta_v,thresholds = Thresholds)
Z = torch.tensor([Data['z'][0:N_steps]]).reshape((N_steps,1)).type(torch.float32)

ro,sp,me = net(Z)

plt.plot(torch.cat(ro,axis=0).detach().numpy(),label='V')
plt.plot(Z.numpy(),label='c')
plt.plot(Data['z'][0:N_steps],label='z')
plt.legend()
