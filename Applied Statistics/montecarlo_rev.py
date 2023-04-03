import numpy as np
import random
import matplotlib.pyplot as plt

def line(p,p1,p2): #p1, p2を通る直線
    x,y = p[0],p[1]
    x1,y1 = p1[0],p1[1]
    x2,y2 = p2[0],p2[1]
    return (y2-y1)*x-(x2-x1)*y-x1*y2+x2*y1

def y(k):
    n = 1000
    cexps = []
    for i in range(n):
        edges = [] #端点を保存
        count = np.zeros(k) #交差している線分を1，非交差を0
        for j in range(k):
            edge_j = [[random.random(),random.random()],[random.random(),random.random()]] #端点を生成
            p1_j,p2_j = edge_j[0],edge_j[1]
            for l in range(j):
                edge_l = edges[l]
                p1_l,p2_l = edge_l[0],edge_l[1]
                if line(p1_j,p1_l,p2_l)*line(p2_j,p1_l,p2_l)<=0 and line(p1_l,p1_j,p2_j)*line(p2_l,p1_j,p2_j)<=0: #交差判定
                    if line(p1_j,p1_l,p2_l)*line(p2_j,p1_l,p2_l)==0 and line(p1_l,p1_j,p2_j)*line(p2_l,p1_j,p2_j)==0:
                        if max(p1_j[0],p2_j[0])<min(p1_l[0],p2_l[0]) or max(p1_l[0],p2_l[0])<min(p1_j[0],p2_j[0]):
                            continue
                        else:
                            count[j]=1
                            count[l]=1
                    else:
                        count[j]=1
                        count[l]=1
                else:
                    continue
            edges.append(edge_j) #端点を追加
        cexps.append(k-np.sum(count)) #非交差の線分の数
    return [np.average(cexps),np.sqrt(np.var(cexps)/n)]

ydata = [y(2), y(5), y(10), y(20), y(50), y(100),y(200)]
yval = [ydata[k][0] for k in range(len(ydata))]
yerr = [ydata[k][1] for k in range(len(ydata))]
x = [2,5,10,20,50,100,200]

print(yval,yerr) #Yについての統計量を表示

fig,ax = plt.subplots() #Y(k)のグラフ
ax.errorbar(x,yval,yerr=yerr,capsize=2,fmt="o",color="black",ecolor="grey")
ax.set_xlim(0,205)
ax.set_xlabel("$k$")
ax.set_ylabel("$Y$")
ax.grid()
ax.legend()
title = "$Y (M={10^3})$"
plt.title(title)
plt.show()