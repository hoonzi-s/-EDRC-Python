#ODE 함수의 parameter fitting
#local optimizing algorithm (quasi-Newton method)

#Ordinary differential equation (ODE) 풀이 기초
#데이터의 시간 간격이 일정하지 않을 때
#Intial point 또한 parameter 값으로 줄 때

from scipy.optimize import minimize
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
def cost(p,x_data, t_data):
    # parameters to be estimated
    a   = p[0]
    p_1 = p[1]
    x0  = p[2]

    x_ = [x0]  # initial x value
    time = t_data[0] #initial time
    for i in range(len(t_data)-1):
        new_start_time = t_data[i]
        new_end_time = t_data[i + 1]
        time = np.append(time, t_data[i + 1])
        x_result = ode_model_3_value1(x_[-1], new_start_time, new_end_time, a, p_1)
        x_ = np.append(x_, x_result[-1])

    Cost=0
    for j in range(len(x_)-1):
        err  = x_data[j] - x_[j]
        Cost = Cost + err**2
    Cost = Cost/len(x_)
    return Cost

#Optimization for parameter estimation (Using local optimizing algorithm)
initial=[2,0.1,0.1]
t_data = np.array([0,   1.2, 2.1,  3.6,  4.2,  5.9])
x_data = np.array([0.1, 0.4, 0.42, 0.45, 0.49, 0.50]) #Data
result = minimize(cost, initial, method='BFGS', args=(x_data, t_data), options={'disp':True})
print("추정된 파라미터 값:", result.x)

#추정된 파라미터로 만든 모델
#각 시간 구간을 나눠서 적분
x_ = [result.x[2]]  # initial x value
time = t_data[0] #initial time
for i in range(len(t_data)-1):
    new_start_time = t_data[i]
    new_end_time   = t_data[i + 1]
    time=np.append(time,t_data[i + 1])
    x_result=ode_model_3_value1(x_[-1], new_start_time, new_end_time,result.x[0], result.x[1])
    x_=np.append(x_, x_result[-1])

#시각화
plt.figure(1)
plt.scatter(time, x_data, c='r', marker='^', label='Data')
plt.plot(time, x_,'bo-', label='Estimated model')
plt.legend(loc='best')
plt.show()