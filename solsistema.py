#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 21:19:46 2022

@author: usuario
"""

# Import modules
import numpy as np
import scipy
import sympy as sym
from scipy import sparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import linalg

from IPython.display import Math
from IPython.display import display

sym.init_printing(use_latex=True)


# matrix build

S=np.diagflat(np.ones(2),1)+np.diagflat(np.ones(2),-1)
A=np.diagflat(-4*np.ones(3))+S
#np.kron(matriz1, matriz2)
I=np.identity(3)

Ma=np.kron(I, A)+np.kron(S,I)

F=np.array([-75,-75,-175,0,0,-100,-50,-50,-150])


u = linalg.solve(Ma, F)

u=np.reshape(u,(3,3))
U=np.zeros([5,5])
U[1:4,1:4]=u

U[:,4]=0
U[:,0]=0
x=np.linspace(0,1,5)
y=np.linspace(0,1,5)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, U)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.clf()