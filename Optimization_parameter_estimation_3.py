#ODE 함수의 parameter fitting
#global optimizing algorithm (differential evolution algorithm)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

#ODE example 2
def ode_model_3(x,t, a, p):
    dxdt = -a*x + p
    return dxdt

#initial value (x_initial)와 start, final time (start_time, end_time)을 바꿔줬을 때 그 사이에서의 x 값을 구하는 함수
def ode_model_3_value1(x_initial, start_time, end_time, a, p_1):
    time = np.linspace(start_time,end_time)
    x=odeint(ode_model_3, x_initial, time, args=(a, p_1))
    return x

#Cost (실험 값과 모델 추정 값의 차이 제곱 합)
def cost(p,x_data):
    a = p[0]
    p_1 =p[1] #parameters to be estimated

    # x_data = np.array([0.1, 0.4, 0.42, 0.45, 0.49, 0.50])  # Data

    x_ = [0.1]  # initial
    time = [0]
    for i in range(5):
        new_start_time = i
        new_end_time = i + 1
        time = np.append(time, i + 1)
        x_result = ode_model_3_value1(x_[-1], new_start_time, new_end_time, a, p_1)
        x_ = np.append(x_, x_result[-1])

    Cost=0
    for j in range(len(x_)):
        err  = x_data[j] - x_[j]
        Cost = Cost + err**2
    Cost = Cost/len(x_)
    return Cost

# Optimization for parameter estimation (Using global optimizing algorithm)
x_data = np.array([0.1, 0.4, 0.42, 0.45, 0.49, 0.50])  # Data
bounds2 = [(0,5), (0, 5)] #boundary
result_2 = differential_evolution(cost,bounds=bounds2,args=(x_data,))
print("추정된 파라미터 값:", result_2.x)

# 추정된 파라미터로 만든 모델
# 각 시간 구간을 나눠서 적분
x_=[0.1]
time=[0]
for i in range(5):
    new_start_time = i
    new_end_time   = i+1
    time=np.append(time,i+1)
    x_result=ode_model_3_value1(x_[-1], new_start_time, new_end_time,result_2.x[0], result_2.x[1])
    x_=np.append(x_, x_result[-1])

#시각화
x_data = np.array([0.1, 0.4, 0.42, 0.45, 0.49, 0.50]) #Data
plt.figure(1)
plt.scatter(time, x_data, c='r', marker='^', label='Data')
plt.plot(time, x_,'bo-', label='Estimated model')
plt.legend(loc='best')
plt.show()