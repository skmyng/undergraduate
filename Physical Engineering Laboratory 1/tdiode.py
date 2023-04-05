# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import pandas as pd

df=pd.read_excel('プロット_1日目.xlsx', sheet_name=2, header=None, index_col=None)

e = 1.602 #*10**(-19)
k = 1.381*10**(-4) #*10**(-19)
t = 298

#tonnel diode
xdata = np.array(df.iloc[:,0]) #voltage of diode [mV]
ydata = np.array(df.iloc[:,1]) #current of resistance [mA]

def current_d(i_s,x): #input: mV, output mA
    return i_s*(np.exp(e*x*10**(-3)/(k*t))-1)

def current_t(i_p,v_p,x): #input: mV, output mA
    return i_p*x/v_p*np.exp(1-x/v_p)

def fitfunc(param,x,y):
    i_s,i_p,v_p = param[0],param[1],param[2]
    residual = y-current_d(i_s,x)-current_t(i_p,v_p,x)
    return residual

param0 = np.array([1.0*10**(-4),50,0.12])
result = optimize.leastsq(fitfunc,param0,args=(xdata,ydata))
i_s,i_p,v_p  = result[0][0],result[0][1],result[0][2]
print('I_s[mA]:', i_s, ',I_p[mA]:', i_p, ',V_p[mV]:', v_p)

xx = np.linspace(-0.1,500,1000)
yy = current_d(i_s,xx)+current_t(i_p,v_p,xx)

fig,ax = plt.subplots()
ax.scatter(xdata*10**(-3),ydata,s=10)
ax.plot(xx*10**(-3),yy,label='fitted',color='black',alpha=0.5,linewidth=1.3)
ax.set_xlim(-0.01,0.5)
ax.set_ylim(-0.05,0.3)
ax.set_xlabel('$V$[V]')
ax.set_ylabel('$I$[mA]')
ax.grid()
ax.legend()
plt.title('Tonnel Diode')
plt.show()
