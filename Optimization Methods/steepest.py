import numpy as np
import matplotlib.pyplot as plt
import time

time_sta = time.perf_counter()

opt = np.array([1,1])
c1 = 0.0001 #parameters
#c2 = 0.9
rho = 0.5
alpha0 = 1
epsilon = 0.0001

def f(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def gradf(x):
    gf0 = 400*x[0]**3+(-400*x[1]+2)*x[0]-2
    gf1 = 200*(x[1]-x[0]**2)
    return np.array([gf0,gf1])

def backtrak(x,d):
    alpha = alpha0
    while f(x+alpha*d) > f(x)+c1*alpha*np.dot(gradf(x),d):
        alpha = rho*alpha
    return alpha

def steepest(x0,maxiter=10000):
    x = np.array(x0)
    sol = [np.array(x)]
    for k in range(maxiter):
        if np.linalg.norm(gradf(x)) > epsilon:
            d = -gradf(x)
            alpha = backtrak(x,d)
            x += alpha*d
            sol.append(np.array(x))
        else:
            break
    return x,sol

def graphs(sol):
    l = len(sol)
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    x = [sol[i][0] for i in range(l)]
    y = [sol[i][1] for i in range(l)]

    xmax,xmin = max(x),min(x)
    ymax,ymin = max(y),min(y)
    xg = np.arange(xmin,xmax,0.001)
    yg = np.arange(ymin,ymax,0.001)
    if len(xg) > 0 and len(yg) > 0:
        X, Y = np.meshgrid(xg,yg)
        Z = f([X,Y])
        p = np.arange(-1,4,1)
        levs = [10.0**p[i] for i in range(len(p))]
        ax1.contour(X,Y,Z,levels=levs,colors="blue")

    ax1.plot(x,y,color="red")
    ax1.set_aspect("equal")
    ax1.set_title('motion of solutions')

    ax2.set_xlabel('iteration')
    ax2.set_ylabel('$||x_{k}-x||$')
    ax2.set_yscale('log')
    dist = [np.linalg.norm(sol[i]-opt) for i in range(l)]
    ax2.plot(np.arange(l),dist)
    ax2.set_title('distance to opt')

    ax3.set_xlabel('iteration')
    ax3.set_ylabel('$f(x)$')
    ax3.set_yscale('log')
    val = [f(sol[i]) for i in range(l)]
    ax3.plot(np.arange(l),val)
    ax3.set_title('f(x)')

    fig.tight_layout()
    plt.show()

def graphs2(sol):
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    x = [sol[i][0] for i in range(100,200)]
    y = [sol[i][1] for i in range(100,200)]
    ax1.plot(x,y,color="red")
    ax1.set_aspect("equal")
    ax1.set_title('solutions of iteration from 100 times to 200 ($x_{0}$)')

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    for j in range(len(sol)):
        if abs(sol[j][0]-opt[0])<0.005 and abs(sol[j][1]-opt[1])<0.005:
            sol = sol[j:]
            break
    for j in range(len(sol)):
        if abs(sol[j][0]-opt[0])<0.004 and abs(sol[j][1]-opt[1])<0.004:
            sol = sol[:j]
            break

    x = [sol[i][0] for i in range(len(sol))]
    y = [sol[i][1] for i in range(len(sol))]

    ax2.plot(x,y,color="red")
    ax2.set_aspect("equal")
    ax2.set_title('solutions close to opt ($x_{0}$)')

    fig.tight_layout()
    plt.show()

def graphs3(sol):
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    x = [sol[i][0] for i in range(100)]
    y = [sol[i][1] for i in range(100)]
    ax1.plot(x,y,color="red")
    ax1.set_aspect("equal")
    ax1.set_title('solutions of iteration from 100 times to 200 ($x_{0}$)')

    fig.tight_layout()
    plt.show()


x0 = np.array([1.2,1.2])
x1 = np.array([-1.2,1])
x2 = np.array([1,-1.2])
x3 = np.array([-1.2,-1.2])

x0,sol0 = steepest(x0)
x1,sol1 = steepest(x1)
x2,sol2 = steepest(x2)
x3,sol3 = steepest(x3)

"""
graphs(sol0)
graphs(sol1)
graphs(sol2)
graphs(sol3)
graphs3(sol1)
"""

print(x0,x1,x2,x3)

time_end = time.perf_counter()

print(time_end-time_sta)
