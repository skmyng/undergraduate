# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import pandas as pd

#constants
e = 1.602 #*10**(-19)
k = 1.381*10**(-4) #*10**(-19)
t = 298
#E_f = 0 (at V_extenal = 0)
E_c = -0.06 #[eV]
E_v = 0.064 #[eV]

#tonnel diode
df=pd.read_excel('プロット_1日目.xlsx', sheet_name=2, header=None, index_col=None)
xdata = np.array(df.iloc[:,0]) #voltage of diode [mV]
ydata = np.array(df.iloc[:,1]) #current of resistance [mA]

def current_d(i_s,x): #input: mV, output mA
    return i_s*(np.exp(e*x*10**(-3)/(k*t))-1)

def f(E): #E[eV]
    return np.exp(-e*E/(k*t))/(1+np.exp(-e*E/(k*t)))

def N_n(E): #E[eV]
    if E > E_c:
        return np.sqrt(E-E_c)
    else:
        return 0

def N_p(E):
    if E < E_v:
        return np.sqrt(E_v-E)
    else:
        return 0

def current_t(a,x): #V[mV]
    y = []
    for i in range(len(x)):
        ans = 0
        n = 100
        xi = x[i]*10**(-3) #[eV]
        dE = (E_v-xi - E_c)/n
        El = np.linspace(E_c,E_v-xi,n+1)
        if E_v-xi > E_c:
            for j in range(n):
                ans += N_n(El[j])*N_p(El[j]+xi)*(f(El[j])-f(El[j]+xi))*dE
            y.append(ans)
        else:
            y.append(ans)
    return a*np.array(y)

def fitfunc(param,x,y):
    i_s,a = param[0],param[1]
    residual = y-current_d(i_s,x)-current_t(a,x)
    return residual

param0 = np.array([1.0*10**(-4),150.0])
result = optimize.leastsq(fitfunc,param0,args=(xdata,ydata))
i_s,a = result[0][0],result[0][1]
print('I_s[mA]:',i_s,',a:',a)

xx = np.linspace(-0.1,200,400)
yy = current_t(a,xx)

xx = np.linspace(-0.1,500,1000)
yy = current_d(i_s,xx)+current_t(a,xx)

fig,ax = plt.subplots()
ax.scatter(xdata*10**(-3),ydata,s=10)
ax.plot(xx*10**(-3),yy,label='fitted by integral',color='black',alpha=0.5,linewidth=1.3)
ax.set_xlim(-0.01,0.5)
ax.set_ylim(-0.05,0.3)
ax.set_xlabel('$V$[V]')
ax.set_ylabel('$I$[mA]')
ax.grid()
ax.legend()
plt.title('Tonnel Diode')
plt.show()
