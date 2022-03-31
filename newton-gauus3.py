#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:12:41 2022

@author: usuario
"""

import  numpy as np
import scipy.linalg as la

C=np.array([[-1.0,0,1.0],[1.0,0.5,0.5],[1.0,-0.5,0.5],[0,1,0.5]])
# the varible C is a matrix
# C[:,0] first column of matrix has  information of 1 cordiante od circles
# C[:,1] Second column of matrix has  information of 2 cordiante od circles  
# C[:,2] third column of matrix has  information of radius of each circle

#

# Error function  construction

def FR(x1,C):
    x=x1[0]
    y=x1[1]
    k=x1[2]
    R=np.sqrt((x-C[:,0])**2+ (y-C[:,1])**2)-(C[:,2]+k)
    return R

# Jacobian error function  construction 
def DFR(x1,C):
    x=x1[0]
    y=x1[1]
    tem=1.0/np.sqrt((x-C[:,0])**2+ (y-C[:,1])**2)
    s1=np.zeros([4,3])
    s1[:,0]=tem*(x-C[:,0])
    s1[:,1]=tem*(y-C[:,1])
    s1[:,2]=-1
    return s1

tol=1e-15
N=2000
x0=np.array([0,0])

def newton_gauss(F,DF,tol, N,x0,C):
    for i in range(N):
        A=DFR(x0,C)
        #F1=FR(x0,C)
        F1=A.T@FR(x0,C)
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
x1=np.array([0,0,0])
    
x=newton_gauss(FR,DFR,1e-1,100,x1,C)
print(x)





