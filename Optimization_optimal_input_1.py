# ODE model의 optimal input design
# local optimizing algorithm (quasi-Newton method)
# time을 1개로 나눴을 때

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
    x_initial=0.01
    x_result = ode_model_3_value1(x_initial, 0,
                                      t_terminal, u)
    Cost = -100*x_result[-1] + u*t_terminal
    return Cost

#Optimization for parameter estimation (Using local optimizing algorithm)
initial=[0.1]
result_1 = minimize(cost, initial, method='SLSQP',
                    options={'disp':True},bounds=((0,0.2),))
print("Optimal input 값은: \n", result_1.x)
print("Optimal output값은: \n",  result_1.fun)

#그래프
time=np.linspace(0,10)
optimal_input=np.ones(np.size(time))*result_1.x
x=ode_model_3_value1(0.01,0,10,result_1.x)
plt.figure(1)
plt.plot(time,x,label='x')
plt.plot(time,optimal_input,label='optimal input')
plt.xlabel('t')
plt.ylabel('x')
plt.title('State & optimal input')
plt.legend(loc='best')
plt.show()
