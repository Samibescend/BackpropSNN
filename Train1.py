# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:40:12 2023

@author: YFGI6212
"""

from pathlib import Path
from Datasets import Signal_Reconstruction
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from Models import Basic

from torch.nn import MSELoss as MSE
from tqdm import tqdm
from matplotlib import pyplot as plt
import plotly as px
import plotly.graph_objs as go

variables = ['z']
Neurons   = 2000
Epochs    = 50
eta = 0.6
mu = 0.0
nu = 0.0
earlyStoppingTresh = 5
pastValuesWindow = 10
numberForecast = 10
""" 
np.random.seed(34567)
torch.manual_seed(0)
 """
### Datasets

data = pd.read_csv(Path("data") / f"VanDerPol-1.csv")
#data = pd.read_csv("./mackey.csv")
#data.rename(columns={"value" : "z"}, inplace= True)

train_data = data[variables][0:95000]
test_data = data[variables][95000:]


dataset = Signal_Reconstruction(window=10,variables=variables, data = train_data)
test_set = Signal_Reconstruction(window=10,variables=variables, data = test_data)

print(len(variables))
ratio = [2*dataset.__len__()//3,dataset.__len__() - 2*dataset.__len__()//3]
train_set, val_set = torch.utils.data.random_split(dataset, ratio)
print(dataset.__getitem__(0))
print(dataset.__getitem__(1))

Train_Data = DataLoader(train_set, 
                        batch_size=50,
                        shuffle=True, 
                        num_workers=1)
print(val_set.__len__())
Validation_Data = DataLoader(val_set, 
                        batch_size=50,
                        shuffle=True, 
                        num_workers=1)

Test_Data   =  DataLoader(test_set, 
                          batch_size=test_set.__len__(),
                        shuffle=False, 
                        num_workers=1)
# Network definition

model = Basic(1, Neurons)

# Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# loss definition before regularization

Loss = MSE()

def train_epoch(epoch_index):
    #global optimizer, Loss, Train_Data, Validation_Data, model, mu, nu
    
    # Training
    model.train()
    for inputs, label in tqdm(iter(Train_Data)):
       
        optimizer.zero_grad()
        xhat, r, spk, mem = model(inputs)
        train_loss = Loss(xhat,inputs)  / (0.001 * spk.sum())
        
        train_loss += mu * torch.linalg.norm(r,ord=1,dim=[1,2]).mean()
        
        train_loss += nu * torch.linalg.norm(r,ord=2,dim=[1,2]).mean()

        #train_loss += eta * torch.linalg.norm(spk,ord=1,dim=[1,2]).mean()
        
        train_loss.backward()
        
        optimizer.step()
        
    # Evaluation
    
    model.eval()
    
    for inputs, label in tqdm(iter(Validation_Data)):
        xhat, r, spk, mem = model(inputs)
        validation_loss = Loss(xhat,inputs)  / (0.001 * spk.sum())
        validation_loss += mu * torch.linalg.norm(r,ord=1,dim=[1,2]).mean()
        validation_loss += nu * torch.linalg.norm(r,ord=2,dim=[1,2]).mean()
        #validation_loss += eta * torch.linalg.norm(spk,ord=1,dim=[1,2]).mean()
   
 
    return(train_loss.item(), validation_loss.item())        



if __name__ == '__main__':
    validation_losses = []
    counter = 0
    for i in range(Epochs):
        train_loss, validation_loss = train_epoch(i)
        print(f"epoch {i}: train: {train_loss}, validation: {validation_loss}")

        validation_losses.append(validation_loss)

        if i == 0:
            minLoss = validation_loss
        else:
            if minLoss > validation_loss:
                minLoss = validation_loss
                torch.save(model.state_dict(), "./epoch.pt")
                counter = 0
            else:
                counter += 1
                if counter == earlyStoppingTresh:
                    print("BREAK")
                    break

    model.load_state_dict(torch.load("./epoch.pt"))


        
    #plt.plot(validation_losses)
    
    
    # show results on the test set
    
    
    IT = Test_Data._get_iterator()
    
    sample, target = IT.__next__()
        
    model.eval()
    
    prediction, r, spk, mem1 = model(sample)
    
    
    np_prediction = []

    for p in (prediction):
        np_prediction.append(p[9].item())

    test_loss = Loss(prediction,target)
    test_loss += mu * torch.linalg.norm(r[0,-1],ord=1).mean()
    test_loss += nu * torch.linalg.norm(r[0,-1],ord=2).mean()
    
    r1 = round(torch.linalg.norm(r[0,-1],ord=1).mean().detach().item(),2)
    r2 = round(torch.linalg.norm(r[0,-1],ord=2).mean().detach().item(),2)
    
       

    

    traceTarget = go.Scatter(
        x = [i for i in range(5000)] ,
        y = target,
        line = go.Line(
        color = "blue"
        )

    )


    tracePredict = go.Scatter(
        x = [i for i in range(5000)] ,
        y = np_prediction,
        line = go.Line(
        color = "green"
        )
    )


    data = go.Data([tracePredict, traceTarget])
    px.offline.plot(data, filename='./Results.html')

    print("Best loss = " + str(min(validation_losses)))
    print("Nombre de spikes " + str(spk.sum()))




















