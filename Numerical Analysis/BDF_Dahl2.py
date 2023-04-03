# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#parameters
initf = 10.0
initg = 0.0
c = -2
et = 10.0
n = 100
dt = et/n
tv = np.linspace(0,et,n)

def f(a):
    return c*a[0]

def BDF(f,init,mtd):
    def BDF1(y,y0,t):
        a = [y,t]
        return y - y0 - dt*f(a)

    def BDF2(y,y0,y1,t):
        a = [y,t]
        return 3/2*y - 2*y0 + 1/2*y1 - dt*f(a)

    def BDF3(y,y0,y1,y2,t):
        a = [y,t]
        return 11/6*y - 3*y0 + 3/2*y1 - 1/3*y2 - dt*f(a)

    def BDF4(y,y0,y1,y2,y3,t):
        a = [y,t]
        return 25/12*y - 4*y0 + 3*y1 - 4/3*y2 + 1/4*y3 - dt*f(a)

    y = np.array([init])
    mtdtxt = str()

    for i in range(n-1):
        if  mtd == 1 or i==0:
            mtdtxt = 'BDF1'
            sol = optimize.root_scalar(lambda x: BDF1(x,y[i],tv[i+1]), x0=y[i],fprime = 1-dt*c,method='newton')
            y = np.append(y, sol.x)

        elif mtd == 2 or i==1:
            mtdtxt = 'BDF2'
            sol = optimize.root_scalar(lambda x: BDF2(x,y[i],y[i-1],tv[i+1]), x0=y[i])
            y = np.append(y, sol.x)

        elif mtd == 3 or i==2:
            mtdtxt = 'BDF3'
            sol = optimize.root_scalar(lambda x: BDF3(x,y[i],y[i-1],y[i-2],tv[i+1]), x0=y[i])
            y = np.append(y, sol.x)

        else:
            mtdtxt = 'BDF4'
            sol = optimize.root_scalar(lambda x: BDF4(x,y[i],y[i-1],y[i-2],y[i-3],tv[i+1]), x0=y[i])
            y = np.append(y, sol.x)

    return y,mtdtxt

def ExpE(f,init):
    lab = 'Explicit Euler'
    y = np.array([init])
    for i in range(n-1):
        y = np.append(y, y[i]+dt*f([y[i],tv[i]]))
    return y,lab

tvf = np.linspace(0,et,100)
eyf = initf*np.exp(c*tvf)
z = c*dt

def draw_graph(dat):
    y,lab = dat[0],dat[1]
    fig,ax = plt.subplots()
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.set_title(lab+'; $n=$'+str(n)+'; $\Delta t=$'+str(dt)+'; $c\Delta t=$'+str(z)[:4])
    ax.plot(tvf,eyf,label='Exact solution')
    ax.plot(tv,y,label=lab)
    ax.legend()
    plt.show()

def order_check():
    global n
    global dt
    global tv
    n = 1
    y1er,y2er,y3er,y4er = [],[],[],[]
    nv = []
    for i in range(10):
        n = n*2
        dt = et/n
        tv = np.linspace(0,et,n)
        nv.append(n)
        y1er.append(BDF(f,initf,1)[0][-1]-eyf[-1])
        y2er.append(BDF(f,initf,2)[0][-1]-eyf[-1])
        y3er.append(BDF(f,initf,3)[0][-1]-eyf[-1])
        y4er.append(BDF(f,initf,4)[0][-1]-eyf[-1])
    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.set_xlabel('$\log_{10} n$',fontsize=8)
    ax1.set_ylabel('$\log_{10}$|error|',fontsize=8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.plot(nv,y1er,label="BDF1")
    ax1.set_title('BDF1'+'; $et=$'+str(et)+'; c='+str(c),fontsize=8)

    ax2.set_xlabel('$\log_{10} n$',fontsize=8)
    ax2.set_ylabel('$\log_{10}$|error|',fontsize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot(nv,y2er,label="BDF2")
    ax2.set_title('BDF2'+'; $et=$'+str(et)+'; c='+str(c),fontsize=8)

    ax3.set_xlabel('$\log_{10} n$',fontsize=8)
    ax3.set_ylabel('$\log_{10}$|error|',fontsize=8)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.plot(nv,y3er,label="BDF3")
    ax3.set_title('BDF3'+'; $et=$'+str(et)+'; c='+str(c),fontsize=8)

    ax4.set_xlabel('$\log_{10} n$',fontsize=8)
    ax4.set_ylabel('$\log_{10}$|error|',fontsize=8)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.plot(nv,y4er,label="BDF4")
    ax4.set_title('BDF4'+'; $et=$'+str(et)+'; c='+str(c),fontsize=8)

    plt.rcParams["font.size"] = 8
    fig.tight_layout()
    plt.show()


#order_check()

draw_graph(BDF(f,initf,1))
"""
draw_graph(BDF(f,initf,2))
draw_graph(BDF(f,initf,3))
draw_graph(BDF(f,initf,4))
"""
