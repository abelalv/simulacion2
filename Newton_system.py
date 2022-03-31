import numpy as np
import scipy.linalg as la


def F(x):
    
   return np.array([x[1]-x[0]**3,x[0]**2+x[1]**2-1])  
def DF(x):
    return np.array([[-3*x[0]**2,1],[2*x[0],2*x[1]]])
   
def newton_sistem(F,DF,tol, N,x0):


    for i in range(N):
        A=DF(x0)
        if abs(la.det(A))  < 10**-9:
            print('El sistema no se puede resolver, porque A es singular,intente otro punto inicial')
            break
        x=la.solve(A,-F(x0))
        if np.linalg.norm(x) < tol:
            break
        x0=x0+x
    return x0


tol=1e-10
N=10
x0=np.array([1,2])
x=newton_sistem(F,DF,1e-1,100,np.array([1,1]))
