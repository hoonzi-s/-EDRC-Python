# ODE model의 optimal input design
# local optimizing algorithm (quasi-Newton method)
# time을 여러로 나눴을 때

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ODE example 2
def ode_model_3(x, t, u):
    dxdt = -x + u
    return dxdt

def ode_model_3_value1(x_initial, start_time, end_time, u):
    time = np.linspace(start_time, end_time)
    x = odeint(ode_model_3, x_initial, time, args=(u,))
    return x

# Cost
def cost(u):
    t_terminal=10
    disc_num=6
    t_data = np.linspace(0, t_terminal,num=disc_num)
    time=[0]
    x_=[0.01]
    input=[]
    for i in range(len(t_data) - 1):
        input=np.append(input,u[i])
        new_start_time = t_data[i]
        new_end_time = t_data[i + 1]
        time = np.append(time, t_data[i + 1])
        x_result = ode_model_3_value1(x_[-1], new_start_time,
                                      new_end_time, input[i])
        x_ = np.append(x_, x_result[-1])
    Cost = -100*x_[-1] + np.sum(input)*t_terminal
    return Cost

#Optimization for parameter estimation (Using local optimizing algorithm)
initial=[0.1, 0.1, 0.1, 0.1, 0.1]
result_1 = minimize(cost, initial, method='SLSQP',
                    options={'disp':True},bounds=((0,0.2),(0,0.2),(0,0.2),(0,0.2),(0,0.2)))
print("Optimal input 값은: \n", result_1.x)
print("Optimal output값은: \n",  result_1.fun)

#그래프
time1=np.linspace(0,2)
time2=np.linspace(2,4)
time3=np.linspace(4,6)
time4=np.linspace(6,8)
time5=np.linspace(8,10)

optimal_input1=np.ones(np.size(time1))*result_1.x[0]
optimal_input2=np.ones(np.size(time2))*result_1.x[1]
optimal_input3=np.ones(np.size(time3))*result_1.x[2]
optimal_input4=np.ones(np.size(time4))*result_1.x[3]
optimal_input5=np.ones(np.size(time5))*result_1.x[4]

x1=ode_model_3_value1(0.01,time1[0],time1[-1],result_1.x[0])
x2=ode_model_3_value1(x1[-1],time2[0],time2[-1],result_1.x[1])
x3=ode_model_3_value1(x2[-1],time3[0],time3[-1],result_1.x[2])
x4=ode_model_3_value1(x3[-1],time4[0],time4[-1],result_1.x[3])
x5=ode_model_3_value1(x4[-1],time5[0],time5[-1],result_1.x[4])

time=np.append(time1, time2)
time=np.append(time, time3)
time=np.append(time, time4)
time=np.append(time, time5)

x=np.append(x1,x2)
x=np.append(x,x3)
x=np.append(x,x4)
x=np.append(x,x5)

optimal_input= np.append(optimal_input1, optimal_input2)
optimal_input= np.append(optimal_input, optimal_input3)
optimal_input= np.append(optimal_input, optimal_input4)
optimal_input = np.append(optimal_input,optimal_input5)

print(time)

plt.figure(1)
plt.plot(time,x,label='x')
plt.plot(time,optimal_input,label='optimal_input(5)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('State & optimal input')
plt.show()