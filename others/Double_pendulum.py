from numpy import sin,cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate

G = 9.8
L1,L2 = map(float, input('Enter L1, L2: ').split())
M1 = 1.0
M2 = 1.0
th1, w1, th2, w2 = map(float, input('Enter th1, w1, th2, w2: ').split())

def derives(state,t):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    del_ = state[0]-state[2]
    den1 = (M1+M2)*L1-M2*L1*cos(del_)**2
    dydx[1] = (-(M2*L1*state[1]**2*sin(del_)*cos(del_)
                +M2*L2*state[3]**2*sin(del_)
                +(M1+M2)*G*sin(state[0])
                -M2*G*sin(state[2])*cos(del_))
                /den1)
    dydx[2] = state[3]
    den2 = den1*L2/L1
    dydx[3] = (((M1+M2)*L1*state[1]**2*sin(del_)
                +M2*L2*state[3]**2*sin(del_)*cos(del_)
                -(M1+M2)*G*sin(state[2])
                +(M1+M2)*G*sin(state[0])*cos(del_))
                /den2)
    return dydx

#create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

state = np.radians([th1, w1, th2, w2])

y = integrate.odeint(derives, state, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-(L1+L2),L1+L2), ylim=(-(L1+L2),L1+L2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
#for showing a timer
time_template = 'time = %.lfs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([],[])
    #initializing the timer
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    #updating the timer
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                                interval=dt*1000, blit=True, init_func=init)
plt.show()
