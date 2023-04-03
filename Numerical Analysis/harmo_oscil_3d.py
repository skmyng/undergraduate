# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from time import perf_counter

init1 = np.array([[1.0,1.0,1.0],[0.5,0.0,-0.5]])
init2  = np.array([[0.5,-np.sqrt(2)/2,0.5],[0.0,0.0,0.0]])
init3  = np.array([[np.sqrt(2)/2,0.0,-np.sqrt(2)/2],[0.0,0.0,0.0]])
init4  = np.array([[0.5,np.sqrt(2)/2,0.5],[0.0,0.0,0.0]])
et = 20.0
n = 50
dt = et/n
tv = np.linspace(0,et,n+1)

def exp_E(init):
    label = 'symp_E'
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        q0 = np.array(q)
        p0 = np.array(p)
        q += p0*dt
        p -= np.array([2*q0[0]-q0[1], -q0[0]+2*q0[1]-q0[2], -q0[1]+2*q0[2]])*dt
        y1.append([q[0],p[0]])
        y2.append([q[1],p[1]])
        y3.append([q[2],p[2]])
    return y1,y2,y3,label

def RK4(init):
    label = 'RK4'
    def f(y0):
        q0,p0 = y0[0],y0[1]
        return np.array([p0, -np.array([2*q0[0]-q0[1], -q0[0]+2*q0[1]-q0[2], -q0[1]+2*q0[2]])])
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        y0 = np.array([q,p])
        k1 = f(y0)
        k2 = f(y0 + k1*dt/2)
        k3 = f(y0 + k2*dt/2)
        k4 = f(y0 + k3*dt)
        q += (k1/6 + k2/3 + k3/3 + k4/6)[0]*dt
        p += (k1/6 + k2/3 + k3/3 + k4/6)[1]*dt
        y1.append([q[0],p[0]])
        y2.append([q[1],p[1]])
        y3.append([q[2],p[2]])
    return y1,y2,y3,label

def BDF(init,mtd): #BDFの準備
    def f(y0):
        q0,p0 = np.array(y0[0:3]),np.array(y0[3:6])
        return np.array([p0[0],p0[1],p0[2], -(2*q0[0]-q0[1]), -(-q0[0]+2*q0[1]-q0[2]), -(-q0[1]+2*q0[2])])
    def BDF1(y,y0): #1次
        return y - y0 - dt*f(y)
    def BDF2(y,y0,y1): #2次
        return 3/2*y - 2*y0 + 1/2*y1 - dt*f(y)
    def BDF3(y,y0,y1,y2): #3次
        return 11/6*y - 3*y0 + 3/2*y1 - 1/3*y2 - dt*f(y)
    def BDF4(y,y0,y1,y2,y3): #4次
        return 25/12*y - 4*y0 + 3*y1 - 4/3*y2 + 1/4*y3 - dt*f(y)
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]] #解を格納
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    mtdtxt = str() #k次のBDFのラベル
    for i in range(n): #BDFの実行
        if  mtd == 1 or i==0:
            mtdtxt = 'BDF1'
            z0 = np.array([y1[i][0],y2[i][0],y3[i][0], y1[i][1],y2[i][1],y3[i][1]])
            sol = root(lambda x: BDF1(x,z0), x0=z0)
            y1.append([sol.x[0],sol.x[3]])
            y2.append([sol.x[1],sol.x[4]])
            y3.append([sol.x[2],sol.x[5]])
        elif mtd == 2 or i==1:
            mtdtxt = 'BDF2'
            z0 = np.array([y1[i][0],y2[i][0],y3[i][0], y1[i][1],y2[i][1],y3[i][1]])
            z1 = np.array([y1[i-1][0],y2[i-1][0],y3[i-1][0], y1[i-1][1],y2[i-1][1],y3[i-1][1]])
            sol = root(lambda x: BDF2(x,z0,z1), x0=z0)
            y1.append([sol.x[0],sol.x[3]])
            y2.append([sol.x[1],sol.x[4]])
            y3.append([sol.x[2],sol.x[5]])
        elif mtd == 3 or i==2:
            mtdtxt = 'BDF3'
            z0 = np.array([y1[i][0],y2[i][0],y3[i][0], y1[i][1],y2[i][1],y3[i][1]])
            z1 = np.array([y1[i-1][0],y2[i-1][0],y3[i-1][0], y1[i-1][1],y2[i-1][1],y3[i-1][1]])
            z2 = np.array([y1[i-2][0],y2[i-2][0],y3[i-2][0], y1[i-2][1],y2[i-2][1],y3[i-2][1]])
            sol = root(lambda x: BDF3(x,z0,z1,z2,), x0=z0)
            y1.append([sol.x[0],sol.x[3]])
            y2.append([sol.x[1],sol.x[4]])
            y3.append([sol.x[2],sol.x[5]])
        else:
            mtdtxt = 'BDF4'
            z0 = np.array([y1[i][0],y2[i][0],y3[i][0], y1[i][1],y2[i][1],y3[i][1]])
            z1 = np.array([y1[i-1][0],y2[i-1][0],y3[i-1][0], y1[i-1][1],y2[i-1][1],y3[i-1][1]])
            z2 = np.array([y1[i-2][0],y2[i-2][0],y3[i-2][0], y1[i-2][1],y2[i-2][1],y3[i-2][1]])
            z3 = np.array([y1[i-3][0],y2[i-3][0],y3[i-3][0], y1[i-3][1],y2[i-3][1],y3[i-3][1]])
            sol = root(lambda x: BDF4(x,z0,z1,z2,z3), x0=z0)
            y1.append([sol.x[0],sol.x[3]])
            y2.append([sol.x[1],sol.x[4]])
            y3.append([sol.x[2],sol.x[5]])
    return y1,y2,y3,mtdtxt #解のリストとラベルを出力


def symp_E(init):
    label = 'symp_E'
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        q += np.array(p*dt)
        p += - np.array([2*q[0]-q[1], -q[0]+2*q[1]-q[2], -q[1]+2*q[2]])*dt
        y1.append([q[0],p[0]])
        y2.append([q[1],p[1]])
        y3.append([q[2],p[2]])
    return y1,y2,y3,label

def SV(init):
    label = 'SV'
    def Hq(q):
        return np.array([2*q[0]-q[1], -q[0]+2*q[1]-q[2], -q[1]+2*q[2]])
    def Hp(p):
        return np.array(p)
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        p += - 1/2*Hq(q)*dt
        q += Hp(p)*dt
        p += - 1/2*Hq(q)*dt
        y1.append([q[0],p[0]])
        y2.append([q[1],p[1]])
        y3.append([q[2],p[2]])
    return y1,y2,y3,label

def Ruth(init):
    label = 'Ruth'
    def Hq(q):
        return np.array([2*q[0]-q[1], -q[0]+2*q[1]-q[2], -q[1]+2*q[2]])
    def Hp(p):
        return np.array(p)
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        q += 7/24*Hp(p)*dt
        p += - 2/3*Hq(q)*dt
        q += 3/4*Hp(p)*dt
        p += - -2/3*Hq(q)*dt
        q += -1/24*Hp(p)*dt
        p += - Hq(q)*dt
        y1.append([q[0],p[0]])
        y2.append([q[1],p[1]])
        y3.append([q[2],p[2]])
    return y1,y2,y3,label

def SS(init):
    label='SS'
    def Hq(q):
        return np.array([2*q[0]-q[1], -q[0]+2*q[1]-q[2], -q[1]+2*q[2]])
    def Hp(p):
        return np.array(p)
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        q += 7/48*Hp(p)*dt
        p += - 1/3*Hq(q)*dt
        q += 3/8*Hp(p)*dt
        p += - -1/3*Hq(q)*dt
        q += -1/48*Hp(p)*dt
        p += - Hq(q)*dt
        q += -1/48*Hp(p)*dt
        p += - -1/3*Hq(q)*dt
        q += 3/8*Hp(p)*dt
        p += - 1/3*Hq(q)*dt
        q += 7/48*Hp(p)*dt
        y1.append([q[0],p[0]])
        y2.append([q[1],p[1]])
        y3.append([q[2],p[2]])
    return y1,y2,y3,label

def GL4(init):
    label = 'GL4'
    def f(y):
        q,p = np.array(y[0:3]),np.array(y[3:6])
        return np.array([p[0],p[1],p[2], -(2*q[0]-q[1]), -(-q[0]+2*q[1]-q[2]), -(-q[1]+2*q[2])])
    def g(X,y0):
        X1,X2 = np.array(X[0:6]),np.array(X[6:12])
        return X - np.concatenate([y0 + dt*(1/4*f(X1)+(3-2*np.sqrt(3))/12*f(X2)), y0 + dt*((3+2*np.sqrt(3))/12*f(X1)+1/4*f(X2))], axis=0)
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        y0 = np.array([y1[-1][0],y2[-1][0],y3[-1][0], y1[-1][1],y2[-1][1],y3[-1][1]])
        X0 = np.concatenate([y0,y0],axis=0)
        sol = root(lambda X: g(X,y0),x0=X0)
        X1,X2 = sol.x[0:6], sol.x[6:12]
        y = y0 + dt*(f(X1)+f(X2))/2
        y1.append([y[0],y[3]])
        y2.append([y[1],y[4]])
        y3.append([y[2],y[5]])
    return y1,y2,y3,label

def disc_grad(init,ind):
    label = 'disc_grad'
    def H(y):
        return (y[0]**2 + (y[0]-y[1])**2 + (y[1]-y[2])**2 + y[2]**2 + y[3]**2 + y[4]**2 + y[5]**2)/2
    def nablaH(y):
        q,p = np.array(y[0:3]),np.array(y[3:6])
        return np.array([2*q[0]-q[1],-q[0]+2*q[1]-q[2],-q[1]+2*q[2], p[0],p[1],p[2]])
    def g(y,y0):
        q,p = np.array(y[0:3]),np.array(y[3:6])
        q0,p0 = np.array(y0[0:3]),np.array(y0[3:6])
        qm,pm = (q+q0)/2,(p+p0)/2
        J = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[-1,0,0,0,0,0],[0,-1,0,0,0,0],[0,0,-1,0,0,0]])
        if ind == 1: #AVF, GL2(陰的中点則（台形則)）
            dg = nablaH((y+y0)/2)
        else: #Gonzalez
            z = (y+y0)/2
            dg = nablaH(z) + (H(y)-H(y0)-np.dot(z,y-y0))/np.dot(y-y0,y-y0) * (y-y0)
        return y - y0 - dt*np.matmul(J,dg)
    q = np.array(init[0])
    p = np.array(init[1])
    y1 = [[q[0],p[0]]]
    y2 = [[q[1],p[1]]]
    y3 = [[q[2],p[2]]]
    for i in range(n):
        y0 = np.array([y1[-1][0],y2[-1][0],y3[-1][0], y1[-1][1],y2[-1][1],y3[-1][1]])
        if ind == 1:
            sol = root(lambda x: g(x,y0),x0=y0)
        else:
            sol = root(lambda x: g(x,y0),x0=y0+np.ones(6)*0.1)
        y1.append([sol.x[0],sol.x[3]])
        y2.append([sol.x[1],sol.x[4]])
        y3.append([sol.x[2],sol.x[5]])
    return y1,y2,y3,label

def exactq(ini):
    if ini == 1:
        return ((2-np.sqrt(2))/2 * np.matmul(np.array([[1/2,-np.sqrt(2)/2,1/2]]).T, np.array([np.cos(np.sqrt(2+np.sqrt(2))*tv)]))
                    + 1/2 * np.matmul(np.array([[np.sqrt(2)/2,0,-np.sqrt(2)/2]]).T, np.array([np.sin(np.sqrt(2)*tv)]))
                    + (2+np.sqrt(2))/2 * np.matmul(np.array([[1/2,np.sqrt(2)/2,1/2]]).T, np.array([np.cos(np.sqrt(2-np.sqrt(2))*tv)])))
    elif ini == 2:
        return np.matmul(np.array([[1/2,-np.sqrt(2)/2,1/2]]).T, np.array([np.cos(np.sqrt(2+np.sqrt(2))*tv)]))
    elif ini == 3:
        return np.matmul(np.array([[np.sqrt(2)/2,0,-np.sqrt(2)/2]]).T, np.array([np.cos(np.sqrt(2)*tv)]))
    elif ini == 4:
        return np.matmul(np.array([[1/2,np.sqrt(2)/2,1/2]]).T, np.array([np.cos(np.sqrt(2-np.sqrt(2))*tv)]))

def H_t(sol):
    q1,p1 = [y[0] for y in sol[0]],[y[1] for y in sol[0]]
    q2,p2 = [y[0] for y in sol[1]],[y[1] for y in sol[1]]
    q3,p3 = [y[0] for y in sol[2]],[y[1] for y in sol[2]]
    q1,p1 = np.array(q1),np.array(p1)
    q2,p2 = np.array(q2),np.array(p2)
    q3,p3 = np.array(q3),np.array(p3)
    T = (p1**2 + p2**2 + p3**2)/2
    U = (q1**2 + (q2-q1)**2 + (q3-q2)**2 + q3**2)/2
    return T+U

def tq_graph(sol):
    q1,q2,q3 = [y[0] for y in sol[0]],[y[0] for y in sol[1]],[y[0] for y in sol[2]]
    lab = sol[3]
    fig,ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('$q$')
    ax.plot(tv,q1,label='$q_1$')
    ax.plot(tv,q2,label='$q_2$')
    ax.plot(tv,q3,label='$q_3$')
    ax.legend()
    plt.show()

def qp_graph(sol):
    q1,p1 = [y[0] for y in sol[0]],[y[1] for y in sol[0]]
    q2,p2 = [y[0] for y in sol[1]],[y[1] for y in sol[1]]
    q3,p3 = [y[0] for y in sol[2]],[y[1] for y in sol[2]]
    fig,ax = plt.subplots()
    ax.set_xlabel('$q$')
    ax.set_ylabel('$p$')
    ax.plot(q1,p1,label='1')
    ax.plot(q2,p2,label='2')
    ax.plot(q3,p3,label='3')
    ax.legend()
    plt.show()

def tH_graph(sol):
    lab = sol[3]
    exactH = np.ones(n+1)*H_t(sol)[0]
    fig,ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('$H$')
    ax.plot(tv,exactH,label="exact $H$")
    ax.plot(tv,H_t(sol),label=lab)
    ax.legend()
    plt.show()

def diff_graph(sol,ini):
    q1,q2,q3 = [y[0] for y in sol[0]],[y[0] for y in sol[1]],[y[0] for y in sol[2]]
    lab = sol[3]
    exact_sol = exactq(ini)
    fig,ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('$\Delta q$')
    ax.plot(tv,q1-exact_sol[0],label='$\Delta q_1$')
    ax.plot(tv,q2-exact_sol[1],label='$\Delta q_2$')
    ax.plot(tv,q3-exact_sol[2],label='$\Delta q_3$')
    ax.legend()
    plt.show()

def calc_error(func,init,ini):
    global n
    global dt
    global tv
    n = 25
    errs = []
    for i in range(7):
        n = n*2
        dt = et/n
        tv = np.linspace(0,et,n+1)
        eq = exactq(ini)
        sol = func(init)
        q = np.array([[y[0] for y in sol[0]],[y[0] for y in sol[1]],[y[0] for y in sol[2]]])
        idx = np.unravel_index(np.argmax(np.abs(q-eq)),eq.shape)
        error = np.abs(q-eq)[idx]
        errs.append(-np.log2(error))
    print(errs)

def H_comp(init): #HamiltonianをSymplectic Euler法，離散勾配法，4次RK法について比較
    sol_SE = symp_E(init)
    sol_DG = disc_grad(init,1)
    sol_RK4 = RK4(init)
    fig,ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('$H$')
    ax.plot(tv,H_t(sol_SE),label="Symplectic Euler")
    ax.plot(tv,H_t(sol_DG),label="Discrete Gradient")
    ax.plot(tv,H_t(sol_RK4),label="RK4")
    ax.set_title('Hamiltonian; $N=50$')
    ax.legend()
    plt.show()

def calc_time(func,init):
    rt =[]
    for i in range(10):
        start = perf_counter()
        func(init)
        end = perf_counter()
        rt.append(end-start)
    rt = np.array(rt)
    print(np.mean(rt),np.std(rt))

ini = 1
init = init1
#tq_graph(exp_E(init))
#tq_graph(RK4(init))
#tq_graph(BDF(init,1))
#tq_graph(BDF(init,2))
#tq_graph(BDF(init,3))
#tq_graph(BDF(init,4))

#tq_graph(symp_E(init))
#tq_graph(SV(init))
#tq_graph(Ruth(init))
#tq_graph(SS(init))
#tq_graph(GL4(init))
#tq_graph(disc_grad(init,1))
#tq_graph(disc_grad(init,2))


#qp_graph(exp_E(init))
#qp_graph(symp_E(init))
#tH_graph(exp_E(init))
#tH_graph(RK4(init))
#tH_graph(BDF(init,1))

#tH_graph(symp_E(init))
#tH_graph(SV(init))
#tH_graph(Ruth(init))
#tH_graph(SS(init))

#tH_graph(disc_grad(init,1))
#tH_graph(disc_grad(init,2))

#diff_graph(RK4(init),ini)
#diff_graph(BDF(init,1),ini)
#diff_graph(BDF(init,2),ini)
#diff_graph(BDF(init,3),ini)
#diff_graph(BDF(init,4),ini)

#diff_graph(symp_E(init),ini)
#diff_graph(SV(init),ini)å
#diff_graph(SS(init),ini)
#diff_graph(GL4(init),ini)
#diff_graph(disc_grad(init,1),ini)

H_comp(init)

def BDF1(init):
    return BDF(init,mtd=1)
def BDF2(init):
    return BDF(init,mtd=2)
def BDF3(init):
    return BDF(init,mtd=3)
def BDF4(init):
    return BDF(init,mtd=4)
def GL2(init):
    return disc_grad(init,1)

'''
calc_error(exp_E,init,ini)
print("exp_E")
calc_error(RK4,init,ini)
print("RK4")
calc_error(BDF1,init,ini)
print("BDF1")
calc_error(BDF2,init,ini)
print("BDF2")
calc_error(BDF3,init,ini)
print("BDF3")
calc_error(BDF4,init,ini)
print("BDF4")
calc_error(symp_E,init,ini)
print("symp_E")
calc_error(SV,init,ini)
print("SV")
calc_error(Ruth,init,ini)
print("Ruth")
calc_error(SS,init,ini)
print("SS")
calc_error(GL2,init,ini)
print("GL2")
calc_error(GL4,init,ini)
print("GL4")
'''
'''
calc_time(exp_E,init)
print("exp_E")
calc_time(RK4,init)
print("RK4")
calc_time(BDF1,init)
print("BDF1")
calc_time(BDF2,init)
print("BDF2")
calc_time(BDF3,init)
print("BDF3")
calc_time(BDF4,init)
print("BDF4")
calc_time(symp_E,init)
print("symp_E")
calc_time(SV,init)
print("SV")
calc_time(Ruth,init)
print("Ruth")
calc_time(SS,init)
print("SS")
calc_time(GL2,init)
print("GL2")
calc_time(GL4,init)
print("GL4")
'''
'''
fig,ax = plt.subplots()
ax.set_xlabel('time')
ax.set_ylabel('q')
exact_sol=exactq(1)
ax.plot(tv,exact_sol[0],label='q1')
ax.plot(tv,exact_sol[1],label='q2')
ax.plot(tv,exact_sol[2],label='q3')
ax.set_title('Exact solution')
ax.legend()
plt.show()
'''
