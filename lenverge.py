#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:52:58 2022

@author: usuario
"""
# Import modules

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
import scipy.linalg as la

'''
#x1 = levenberg_marquardt_method(data, c, 50, 1200)
f = lambda t, c1, c2, c3 : c1 * np.exp( -c2 * np.power(t - c3, 2) )
X = np.linspace(0, 5, 100)
Y = f(X, *x)

plt.plot(X, Y, color='cyan')
plt.plot(data[:,0], data[:,1], linestyle='', markersize=8, marker='.', color='blue')
plt.show()


'''


Data=np.array([[1,3],[2,5],[2,7],[3,5],[4,1]])
x0 = np.array([1, 1, 1])
def FR(x,Data):
    Y=x[0]*np.exp(-x[1]*(Data[:,0]-x[2])**2)-Data[:,1]
    return Y

# Jacobian error function  construction 
def DFR(x,Data):
    n=len(Data)
    s1=np.zeros([n,3])
    tem=np.exp(-x[1]*(Data[:,0]-x[2])**2)
    s1[:,0]=tem
    s1[:,1]=tem*(-x[0]*(Data[:,0]-x[2])**2)
    s1[:,2]=tem*(2.0*x[0]*x[1]*(Data[:,0]-x[2]))
    return s1




def newton_gauss(F,DF,tol, N,x0,C):
   for i in range(N):
        A=DFR(x0,C)
        F1=A.T@C[:,1]
        A=A.T@A
        if abs(la.det(A))  < 10**-9:
            print('El sistema no se puede resolver, porque A es singular,intente otro punto inicial')
            break
        q, r = np.linalg.qr(A)
        S1=q.T@F1
        x=la.solve(r,-S1)
        x0=x0+x
        if np.linalg.norm(x)/np.linalg.norm(x0) < tol:
            break
        
        return x0
    
# initial point
tol=1e-15
N=10000
x0=np.array([1,1,1])
C=Data
#x=newton_gauss(FR,DFR,tol,N,x0,Data)
tetha=50
#x=levenberg_marquardt_method(FR,DFR,tol, N,x0,Data,tetha)
#print(x)
 
#def levenberg_marquardt_method(F,DF,tol, N,x0,C,tetha):
for i in range(N):

    A=DFR(x0,C)
    F1=A.T@FR(x0,C)
    A=A.T@A
    Lambda=tetha*np.diagflat(np.diagonal(A))
    A=A+Lambda
     
    if abs(la.det(A))  < 10**-9:
            print('El sistema no se puede resolver, porque A es singular,intente otro punto inicial')
            break
        
    x=la.solve(A,-F1)
 
    x0=x0+x
        
    if np.linalg.norm(x)/np.linalg.norm(x0) < tol:
        print('solución con existo')
        break
    #return i

f = lambda t,x : x[0] * np.exp( -x[1] * np.power(t - x[2], 2) )
    
t = np.linspace(0, 5, 100)
Y =f(t,x0)
plt.plot(t, Y, color='cyan')
plt.plot(Data[:,0], Data[:,1], linestyle='', markersize=8, marker='.', color='blue')
plt.show()

