# _*_ coding: utf-8 _*_
import numpy as np

def PM(A):
    A = np.array(A,dtype=np.float64)
    maxiter = 10000
    x = np.ones(len(A)) #初期ベクトル
    for i in range(maxiter):
        former_x = x
        y = np.dot(A,former_x)
        x = y/np.linalg.norm(y)
        if np.linalg.norm(x-former_x)<1.0/1000:
            m = np.argmax(abs(x)) #絶対値最大の成分のインデックスを取得
            return y[m]/former_x[m] #成分比として固有値を計算
            break
    else:
        return "no convergence"

A1 = np.array(
        [[10,0,0],
        [0,3,4],
        [0,-2,2]])

A2 = np.array(
        [[1,0,0],
        [0,2,-1],
        [0,1,-2]])

A3 = np.array(
        [[2,0,0],
        [0,2,2],
        [0,3,1]])

print(PM(A1))
print(PM(A2))
print(PM(A3))
