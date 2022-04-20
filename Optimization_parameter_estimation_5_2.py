#ODE 함수의 parameter fitting
#global optimizing algorithm (differential evolution algorithm)

#Ordinary differential equation (ODE) 풀이 기초
#데이터의 시간 간격이 일정하지 않을 때

#파라미터 값이 굉장히 작을 때 + log10 형태로 파라미터 치환

from scipy.optimize import differential_evolution
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#ODE example 2
def ode_model_3(x,t, k1_, k2_):
    alpha = 126.2
    beta  = 91.9
    dxdt = 10**k1_*(alpha-x)*(beta-x)**2 -10**k2_*x**2
    return dxdt

#initial value (x_initial)와 start, final time (start_time, end_time)을 바꿔줬을 때 그 사이에서의 x 값을 구하는 함수
def ode_model_3_value1(x_initial, start_time, end_time, k1_, k2_):
    time = np.linspace(start_time,end_time)
    x=odeint(ode_model_3, x_initial, time, args=(k1_, k2_))
    return x

#Cost (실험 값과 모델 추정 값의 차이 제곱 합)
def cost(p, x_data, t_data):
    k1_ = p[0]
    k2_ = p[1] #parameters to be estimated

    # t_data = np.array([0, 1.2, 2.1, 3.6, 4.2, 5.9])
    # x_data = np.array([0.1, 0.4, 0.42, 0.45, 0.49, 0.50])  # Data

    x_ = [0]  # initial x value
    time = t_data[0] #initial time
    for i in range(len(t_data)-1):
        new_start_time = t_data[i]
        new_end_time = t_data[i + 1]
        time = np.append(time, t_data[i + 1])
        x_result = ode_model_3_value1(x_[-1], new_start_time, new_end_time, k1_, k2_)
        x_ = np.append(x_, x_result[-1])

    Cost=0
    for j in range(len(x_)-1):
        err  = x_data[j] - x_[j]
        Cost = Cost + err**2
    Cost = Cost/len(x_)
    return Cost

#Optimization for parameter estimation (Using local optimizing algorithm)
t_data = np.array([0,1,2,3,4,5,6,7,9,11,14,19,24,29,39])
x_data = np.array([0,1.4,6.3,10.5,14.2,17.6,21.4,23.0,27.0,30.5,34.4,38.8,41.6,43.5,45.3])  # Data
bounds2 = [(-10,10), (-10, 10)] #boundary
result_2 = differential_evolution(cost,bounds=bounds2,args=(x_data,t_data))
print("추정된 파라미터 값:", result_2.x)

#추정된 파라미터로 만든 모델
#각 시간 구간을 나눠서 적분
x_ = [0]  # initial x value
time = t_data[0] #initial time
for i in range(len(t_data)-1):
    new_start_time = t_data[i]
    new_end_time   = t_data[i + 1]
    time=np.append(time,t_data[i + 1])
    x_result=ode_model_3_value1(x_[-1], new_start_time, new_end_time,result_2.x[0], result_2.x[1])
    x_=np.append(x_, x_result[-1])

#시각화
plt.figure(1)
plt.scatter(time, x_data, c='r', marker='^', label='Data')
plt.plot(time, x_,'bo-', label='Estimated model')
plt.legend(loc='best')
plt.show()