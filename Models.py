# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:48:07 2023

@author: YFGI6212
"""

import torch
import torch.nn as nn

import snntorch as snn



class Basic(nn.Module):
    def __init__(self,input_size,num_hidden,beta=0.6):
        
        super().__init__()
        print(input_size)
        self._input_size = input_size
        self._num_hidden = num_hidden
        self._beta = beta
        
        self.input_layer = nn.Linear(self._input_size,num_hidden)
        
        
        
        self.lif1 = snn.RLeaky(self._beta,
                               V = 0.2,
                               all_to_all = True,
                               linear_features=self._num_hidden,
                               init_hidden=True,
                               reset_mechanism = "subtract",
                               learn_beta = True,
                               learn_threshold=True,
                               learn_recurrent=True,
                               output = True)
        
        
        self.readout = snn.Leaky(self._beta,
                              init_hidden=True,
                              output = True,
                              reset_mechanism = 'none',
                              learn_beta = False,
                              )
        
        self.output = nn.Linear(self._num_hidden,self._input_size,bias=True)
        
        

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
        n_batch = x.shape[0]
        n_steps = 10
        n_var   = x.shape[2]
        
        """  print("nbatch")
        print(n_batch)
        print("nsteps")
        print(n_steps)
        print("nvar")
        print(n_var)
        """


        spk_lst    = []
        r_lst      = []
        xhat_lst   = []
        mem1_lst   = []
        
        
        # Initalize membrane potential
        mem_1 = self.lif1.init_rleaky()
        mem_2 = self.readout.init_leaky()
        self.readout.reset_hidden()
        self.lif1.reset_hidden()
        
        for step in range(n_steps):
            y = self.input_layer(x[:,step])    # c(t) = input_layer(x(t))

            spk1, mem1 = self.lif1(y)         # spk_1 = lif1(c(t),spk(t))
            spk_lst.append(spk1)
            mem1_lst.append(mem1)
            
            spk2, r  = self.readout(spk1) # r(t) = lif_2(spk_1)
            r_lst.append(r)
            
            xhat = self.output(r)       # x_hat = readout(r(t))
            xhat_lst.append(xhat)
            
            # transform lists to tensors
        
        

        xhat_lst = torch.cat([xhat_lst[i].reshape((n_batch,1,n_var)) for i in range(n_steps)],axis=1)
        r_lst = torch.cat([r_lst[i].reshape((n_batch,1,self._num_hidden)) for i in range(n_steps)],axis=1)
        spk_lst = torch.cat([spk_lst[i].reshape((n_batch,1,self._num_hidden)) for i in range(n_steps)],axis=1)
        mem1_lst = torch.cat([mem1_lst[i].reshape((n_batch,1,self._num_hidden)) for i in range(n_steps)],axis=1)
        
            
        return(xhat_lst, r_lst, spk_lst, mem1_lst)