# _*_ coding: utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt

def CG(A,b):
    n = len(b)
    x = np.zeros(n)
    r = b-np.dot(A,x)
    p = np.array(r)
    re = [] #グラフ作成用に残差を記録
    maxiter = 1000
    for k in range(maxiter):
        Ap = np.dot(A,p) #Apを予め計算
        alpha = np.dot(r,p)/np.dot(p,Ap)
        x += alpha*p #x,rを更新
        r += -alpha*np.dot(A,p)
        beta = -np.dot(r,Ap)/np.dot(p,Ap)
        p = r+beta*p #pを更新
        re.append(np.linalg.norm(b-np.dot(A,x)))
        if np.linalg.norm(b-np.dot(A,x))<1.0/10000*np.linalg.norm(b):
            return x,re
            break
    else:
        return "no convergence"

def CGgraph(re):
    x = np.arange(1,len(re)+1)
    plt.xlabel('iteration')
    plt.ylabel('$|| b-Ax^{(k)} ||$')
    plt.plot(x,re)
    plt.show()

A = np.array([[2,1,2],[1,3,-1],[2,-1,4]])
b = np.array([1,4,2])
print(CG(A,b)[0])
print(CGgraph(CG(A,b)[1]))

def laplace(n):
    A = np.zeros([n,n])
    A[0,0] = -2
    A[0,1] = 1
    A[n-1,n-2] = 1
    A[n-1,n-1] = -2
    for i in range(1,n-1):
        A[i,i-1] = 1
        A[i,i] = -2
        A[i,i+1] = 1
    return A

A = laplace(10)
b = np.zeros(10)
b[0] = 1
b[10-1] = 2
print(CG(A,b)[0])
print(CGgraph(CG(A,b)[1]))

A = laplace(100)
b = np.zeros(100)
b[0] = 1
b[100-1] = 2
print(CG(A,b)[0])
print(CGgraph(CG(A,b)[1]))

A = laplace(1000)
b = np.zeros(1000)
b[0] = 1
b[1000-1] = 2
print(CG(A,b)[0])
print(CGgraph(CG(A,b)[1]))
