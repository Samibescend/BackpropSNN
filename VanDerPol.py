# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:50:36 2023

@author: YFGI6212
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

def model(z,t):
    global alpha, epsilon
    
    dxdt = z[1]
    dydt = - z[0] + epsilon * ( 1 - z[0]*z[0])*z[1]
    dzdt = alpha * z[2] + z[1]
    return([dxdt,dydt,dzdt])

z0 = [0.2,0.0,0.0]
epsilon = 5.2
alpha = -0.8

# time points
Tmax = 1000
t = np.linspace(0,Tmax,num=Tmax*100)

# solve ODE
z = odeint(model,z0,t)

# plot results
figure, axis = plt.subplots(4, 1)
axis[0].plot(z[:,0],z[:,1])
#axis[0].set_title("Full dynamics")

time = [i for i in range(t.shape[0])]
axis[1].plot(time,z[:,0])
axis[1].set_title(r"$x(t)$")

axis[2].plot(time,z[:,1])
#axis[2].set_title(r"$y(t)$")
axis[3].plot(time,z[:,2])

#plt.plot(t,z[:,1],'r--',label=r'$\frac{dy}{dt}=-y+3$')
#plt.ylabel('response')
#plt.xlabel('time')
#plt.legend(loc='best')
plt.show()

data = pd.DataFrame(z,columns=['x','y','z'])
data.to_csv('data/VanDerPol-1.csv')
