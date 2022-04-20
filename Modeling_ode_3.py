#Ordinary differential equation (ODE) 풀이 기초
#Input design 2 (discrete)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#ODE example 2
def ode_model_2(x,t):
    a = 2
    u = input_2(t)
    dxdt = -a*x + u
    return dxdt

#Input design
def input_2(t):
    if t < 2.5:
        u = 0.1
    else:
        u = 0.5
    return u

#Initial value
x_initial=1

#time range
t = np.linspace(0,5)

#ODE solve
x = odeint(ode_model_2, x_initial, t)
u_2 = []
for i in t:
    u_2=np.append(u_2,input_2(i))

print(u_2[0])

#시각화
plt.figure(1)
plt.plot(t,x, label='state, x(t)')
plt.plot(t,u_2, label='input, u_1(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('x')
plt.title('t vs x, Input')
plt.show()