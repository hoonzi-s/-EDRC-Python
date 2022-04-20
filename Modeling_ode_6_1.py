#Ordinary differential equation (ODE) 풀이 기초

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#ODE example 1
def ode_model3(x,t,p):
    a1 = 0.1
    a2 = 0.2
    x1 = x[0]
    x2 = x[1]
    dx1dt = -a1*x2 + p
    dx2dt = -a2*x1 + p
    dxdt = [dx1dt, dx2dt]
    return dxdt

#External parameter
p=1

#initial value (x_initial)와 start, final time (start_time, end_time)을 바꿔줬을 때 그 사이에서의 x 값을 구하는 함수
def ode_model_3_value1(x_initial, start_time, end_time):
    time = np.linspace(start_time,end_time)
    x=odeint(ode_model3, x_initial, time, args=(p,))
    return x

# #ODE solve
# x = odeint(ode_model, x_initial, t, args=(1,))
# x1 = x[:,0]
# x2 = x[:,1]

#각 시간 구간을 나눠서 적분
x_=[[3, 4]] #Initial value
time_data= [0, 1.5, 2, 4, 4.2, 7]
time=time_data[0]
for i in range(5):
    new_start_time = time_data[i]
    new_end_time   = time_data[i+1]
    time           = np.append(time,time_data[i+1])
    x_result       = ode_model_3_value1(x_[-1], new_start_time, new_end_time)
    x_             = np.vstack((x_, x_result[-1]))

x1 = x_[:,0]
x2 = x_[:,1]

#시각화
plt.figure(1)
plt.plot(time, x1, 'bo-',label='x1')
plt.plot(time, x2, 'ro-',label='x2')
plt.legend(loc='best')
plt.show()