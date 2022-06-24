# Project:
#   VQA
# Description:
#   Script for plotting the gradients for loss function term fcn2
# Author: 
#   Sergio Tascon-Morales


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from torch import zeros_like

GAMMA = 2



x,y = np.meshgrid(np.linspace(0,5,20),np.linspace(0,3,20))
z = x*(GAMMA-y)
z[np.where(z<0)] = 0 #RELU


u1 = np.zeros_like(x).ravel()
v1 = np.zeros_like(y).ravel()
cnt = 0
for (x_,y_) in zip(x.ravel(),y.ravel()):
    if x_*(GAMMA - y_) >= 0 and y_<= GAMMA:
        u1[cnt] = GAMMA - y_
        v1[cnt] = -x_
    cnt+=1

u = GAMMA - y
v = -x

#plt.contour(y,x,z, 20)
#plt.colorbar();
#plt.quiver(y,x,-v1,-u1)

plt.quiver(y,x,-v1/(np.sqrt(v1**2 + u1**2)),-u1/(np.sqrt(v1**2 + u1**2)))
plt.grid()
plt.show()

""" 
x,y = np.meshgrid(np.linspace(0,5,20),np.linspace(0,3,20))
z = 2*x+1*y -x*y
z[np.where(z<0)] = 0 #RELU


u1 = np.zeros_like(x).ravel()
v1 = np.zeros_like(y).ravel()
cnt = 0
for (x_,y_) in zip(x.ravel(),y.ravel()):
    if 2*x_ + 1*y_ - x_*y_ >= 0 and y_<= GAMMA:
        u1[cnt] = 2 - y_
        v1[cnt] = 1-x_
    cnt+=1


#plt.contour(y,x,z, 20)
#plt.colorbar();
plt.quiver(y,x,-v1,-u1)

#plt.quiver(y,x,-v1/(np.sqrt(v1**2 + u1**2)),-u1/(np.sqrt(v1**2 + u1**2)))
plt.grid()
plt.show()
"""