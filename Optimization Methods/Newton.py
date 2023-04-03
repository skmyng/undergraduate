import numpy as np
import matplotlib.pyplot as plt

opt = np.array([1,1])
epsilon = 0.0001

def f(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def gradf(x):
    gf0 = 400*x[0]**3+(-400*x[1]+2)*x[0]-2
    gf1 = 200*(x[1]-x[0]**2)
    return np.array([gf0,gf1])

def hessef(x):
    hf00 = 1200*x[0]**2+(-400*x[1]+2)
    hf01 = -400*x[0]
    hf10 = -400*x[0]
    hf11 = 200
    return np.array([[hf00,hf01],[hf10,hf11]])

def Newton(x0,maxiter=10000):
    x = np.array(x0)
    sol = [np.array(x)]
    for i in range(maxiter):
        if np.linalg.norm(gradf(x)) > epsilon:
            d = -np.dot(np.linalg.inv(hessef(x)),gradf(x))
            x += d
            sol.append(np.array(x))
        else:
            break
    return x,sol

def grafs(sol):
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

def grafs2(sol0,sol1):
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    solset = [sol0,sol1]
    for i in range(len(solset)):
        for j in range(len(solset[i])):
            if abs(solset[i][j][0]-1)<0.005 and abs(solset[i][j][1]-1)<0.005:
                solset[i] = solset[i][j:]
                break
        for j in range(len(solset[i])):
            if abs(solset[i][j][0]-1)<0.004 and abs(solset[i][j][1]-1)<0.004:
                solset[i] = solset[i][:j]
                break

    x = [solset[0][i][0] for i in range(len(solset[0]))]
    y = [solset[0][i][1] for i in range(len(solset[0]))]
    ax1.plot(x,y,color="red")
    ax1.set_aspect("equal")
    ax1.set_title('motion of solutions ($x_{0}$)')

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    x = [solset[1][i][0] for i in range(len(solset[1]))]
    y = [solset[1][i][1] for i in range(len(solset[1]))]
    ax2.plot(x,y,color="red")
    ax2.set_aspect("equal")
    ax2.set_title('motion of solutions ($x_{1}$)')

    fig.tight_layout()
    plt.show()

x0 = np.array([1.2,1.2])
x1 = np.array([-1.2,1])
x2 = np.array([1,-1.2])
x3 = np.array([-1.2,-1.2])

x0,sol0 = Newton(x0)
x1,sol1 = Newton(x1)
x2,sol2 = Newton(x2)
x3,sol3 = Newton(x3)

grafs(sol0)
"""
grafs(sol1)
grafs(sol2)
grafs(sol3)
grafs2(sol0,sol1)
"""
print(x0,x1,x2,x3)
