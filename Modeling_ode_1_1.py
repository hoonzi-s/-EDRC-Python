#Ordinary differential equation (ODE) 풀이 기초
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
    dx1dt = -a1*x2 + 1
    dx2dt = -a2*x1 + 2
    dxdt = [dx1dt, dx2dt]
    return dxdt

#Initial value
x_initial=[5, 5]

#time range
t = np.linspace(0,10)

#ODE solve
x = odeint(ode_model, x_initial, t)
x1 = x[:,0]
x2 = x[:,1]

#시각화
plt.figure(1)
plt.plot(t,x1,label='x1')
plt.plot(t,x2,label='x2')
plt.legend(loc='best')
plt.show()