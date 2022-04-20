#Ordinary differential equation (ODE) 풀이 기초
#Input design 2 (discrete)
#Multivariate

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#ODE example 1
def ode_model(x,t):
    a1 = 0.1
    a2 = 0.2
    x1 = x[0]
    x2 = x[1]
    u = input_1(t)
    dx1dt = -a1*x2 + u
    dx2dt = -a2*x1 + u
    dxdt = [dx1dt, dx2dt]
    return dxdt

#Input design
def input_1(t):
    if t < 2.5:
        u = 0
    else:
        u = 5
    return u

#Initial value
x_initial=[5, 5]

#time range
t = np.linspace(0,10)

#ODE solve
x = odeint(ode_model, x_initial, t)
x1 = x[:,0]
x2 = x[:,1]

#input
u_1 = []
for i in t:
    u_1=np.append(u_1,input_1(i))

#시각화
plt.figure(1)
plt.plot(t,x1,label='x1')
plt.plot(t,x2,label='x2')
plt.plot(t,u_1,label='input')
plt.legend(loc='best')
plt.show()