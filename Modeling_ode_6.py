#Ordinary differential equation (ODE) 풀이 기초


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#ODE example 2
def ode_model_3(x,t,p):
    a = 2 #internal parameter
    dxdt = -a*x + p
    return dxdt

#External parameter
p_1=1
p_2=1.1

#initial value (x_initial)와 start, final time (start_time, end_time)을 바꿔줬을 때 그 사이에서의 x 값을 구하는 함수
def ode_model_3_value1(x_initial, start_time, end_time):
    time = np.linspace(start_time,end_time)
    x=odeint(ode_model_3, x_initial, time, args=(p_1,))
    return x

#각 시간 구간을 나눠서 적분
time_data= [0, 1.5, 2, 4, 4.2, 7]
x_       = [0.1]
time     = time_data[0]
for i in range(len(time_data)-1):
    new_start_time = time_data[i]
    new_end_time   = time_data[i+1]
    time=np.append(time,time_data[i+1])
    x_result=ode_model_3_value1(x_[-1], new_start_time, new_end_time)
    x_=np.append(x_, x_result[-1])

#시각화
plt.figure(1)
plt.plot(time, x_,'bo-')
plt.show()