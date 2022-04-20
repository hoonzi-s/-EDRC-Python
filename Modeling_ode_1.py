#Ordinary differential equation (ODE) 풀이 기초

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#ODE example 1
def ode_model1(x,t):
    a=2
    dxdt = -a*x
    return dxdt

#Initial value
x_initial=1

#time range
t = np.linspace(0,10)

#ODE solve
x = odeint(ode_model1, x_initial, t)

#시각화
plt.figure(1)
plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x')
plt.title('t vs x')
plt.show()


