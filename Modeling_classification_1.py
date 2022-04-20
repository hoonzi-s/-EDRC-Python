#A type과 B type 분류하기

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

data = np.array([[1, 0],
		        [2, 0],
		        [3, 0],
		        [4, 0],
		        [5, 0],
		        [6, 1],
		        [7, 1],
		        [8, 1],
		        [9, 1],
		        [10, 1]])

#데이터 분리
x_data = data[:,0].reshape(-1,1) #row data 형태로 만들어 줌.
y_data = data[:,1].reshape(-1,1) #row data 형태로 만들어 줌.

#시각화
plt.plot(x_data,y_data,'bo',label='data')
plt.legend(loc='upper left')
plt.show()

x_data_mod = np.insert(x_data, 0, 1, axis=1) #상수 값 부분을 x 축 벡터에 합쳐서 표현

#Sigmoid function
def Sigmoid_fun(x_data_mod, p):
    p = np.array(p)
    lin_val = np.matmul(x_data_mod, p)
    return 1/(1+np.exp(-lin_val))

p_example=np.array([[0.1],[0.1]])
print("Sigmoid function value", Sigmoid_fun(x_data_mod, p_example))

#Logistic regression cost
def Logistic_regression_cost(x_data_mod,y_data,p):
    return -np.matmul(y_data.T, np.log(Sigmoid_fun(x_data_mod,p))) - np.matmul((1-y_data).T, np.log(1-Sigmoid_fun(x_data_mod,p)))

print("Sigmoid cost value by p_example", Logistic_regression_cost(x_data_mod,y_data,p_example))

def Calculate_new_parameter(x_data_mod, y_data, p):
	p=np.array(p)
	step_size = 0.01
	N = np.size(x_data_mod, 0)
	Gradient = np.matmul(x_data_mod[:,1].T, Sigmoid_fun(x_data_mod, p) - y_data)/N
	p_new = p.T - step_size*Gradient
	p_new = p_new.T
	return p_new

#inital guess
p_new = np.array([[-1.1],[5]])
cost = Logistic_regression_cost(x_data_mod, y_data, p_new)

#iteration
for i in range(500):
	p_new_new = Calculate_new_parameter(x_data_mod, y_data, p_new[:, i].reshape(-1,1))
	p_new = np.hstack((p_new, p_new_new))
	cost_new = Logistic_regression_cost(x_data_mod, y_data, p_new_new)
	cost = np.vstack((cost, cost_new))
	print(cost_new)

y_predict_numerical = Sigmoid_fun(x_data_mod, p_new[:,499])
print("Numerical prediction: ", y_predict_numerical)
print("Parameters: ", p_new[:,499])

xx=np.linspace(-10,10,num=21).reshape(-1,1)
xx_mod = np.insert(xx, 0, 1, axis=1) #상수 값 부분을 x 축 벡터에 합쳐서 표현
yy=Sigmoid_fun(xx_mod,p_new[:,499])

plt.figure(1)
plt.plot(xx, yy)
plt.show()

#Figure
plt.figure(4)
plt.subplot(2,1,1)
plt.scatter(x_data, y_data, label='data')
plt.plot(x_data, y_predict_numerical, 'r-', label='predict_numerical')
plt.legend(loc='upper left')

plt.subplot(2,1,2)
plt.plot(cost, 'r-', label='predict_numerical')
plt.show()