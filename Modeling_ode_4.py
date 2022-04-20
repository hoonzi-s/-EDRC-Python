#Ordinary differential equation (ODE) 풀이 기초
#Parameter

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

#Initial value
x_initial=1

#time range
t = np.linspace(0,5)

#ODE solve
x_1 = odeint(ode_model_3, x_initial, t, args=(p_1,))
x_2 = odeint(ode_model_3, x_initial, t, args=(p_2,))

#시각화
plt.figure(1)
plt.plot(t,x_1, label='state, x_1(t)')
plt.plot(t,x_2, label='state, x_2(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('x')
plt.title('t vs x, Different parameters')
plt.show()

#Parameter 값 (p)를 바꿔줬을 때 t=0, 1, 2, 3, 4, 5에서의 x 값을 구하는 함수
def ode_model_3_value(p):
    time=np.linspace(0,5,6)
    x=odeint(ode_model_3, x_initial, time, args=(p,))
    return x

print("Parameter 값이 1일 때 t=[0,1,2,3,4,5]에서의 x 값: \n", ode_model_3_value(1))
print("Parameter 값이 1.1일 때 t=[0,1,2,3,4,5]에서의 x 값: \n", ode_model_3_value(1.1))