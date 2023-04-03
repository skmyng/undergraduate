# _*_ coding: utf-8 _*_
import numpy as np

def LU(A):
    A = np.array(A,dtype=np.float64)
    n = len(A)
    p = np.empty(n)
    for i in range(n):
        p[i] = i
    for k in range(n-1):
        m = k
        for l in range(k,n): #枢軸選択、[k,n-1]にあるmのうち|A[l,k]|が最大になるmを探す
            if abs(A[l,k]) > abs(A[m,k]):
                m = l
        p_m,p_k = p[m],p[k] #p[m],p[k]の交換
        p[m],p[k] = p_k,p_m
        A_m,A_k = np.array(A[m]),np.array(A[k]) #A[m],A[k]の交換
        A[m],A[k] = A_k,A_m
        w = 1.0/A[k,k]
        for i in range(k+1,n):
            A[i,k] = A[i,k]*w #PAの狭義下三角部分を作成
            for j in range(k+1,n): #前進消去
                A[i,j] -= A[i,k]*A[k,j]
        print(A)
    L=np.zeros([n,n])
    U=np.zeros([n,n])
    for i in range(n-1): #L,Uを作成
        L[i,i] = 1
        U[i,i] = A[i,i]
        for j in range(i+1,n):
            L[j,i] = A[j,i]
            U[i,j] = A[i,j]
    L[n-1,n-1] = 1
    U[n-1,n-1] = A[n-1,n-1]
    return p,L,U

def LU_check(A): #PA,LUを出力する関数
    p,L,U = LU(A)
    n = len(A)
    PA = np.empty([n,n])
    for i in range(n):
        PA[int(p[i])] = A[i]
    return (PA,np.dot(L,U))

A = np.array(
    [[10,2,3,-1],
    [2,10,-2,0],
    [-1,3,10,4],
    [2,-1,5,10]])

print(LU(A))
print(LU_check(A))
