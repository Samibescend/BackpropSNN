# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:32:21 2023

@author: YFGI6212
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader



class Signal_Reconstruction(Dataset):
    """ single signal reconstruction """
    def __init__(self, csv_file="VanDerPol-1", variables=['z'],window = 1, root_dir = "data", data = None, test = False):
        
        if data is None:
            self._data = pd.read_csv(Path(root_dir) / f"{csv_file}.csv")
        else:
            self._data = data
        self._data = self._data[variables]
        self._window = window
        self._nvars  = len(variables)
        
    def __len__(self):
        return self._data.shape[0]-self._window+1
    
    def __getitem__(self, idx):
        if idx >= len(self._data["z"].values) - (self._window * 2):
            idx = len(self._data["z"].values) - (self._window * 2)
        if self._window > 4000:
            print("TEST")
        return (torch.tensor(self._data["z"].values[idx:idx+self._window].reshape((self._window,self._nvars))).type(torch.float32),
                torch.tensor(self._data["z"].values[idx+self._window + 1]).type(torch.float32)
                )



