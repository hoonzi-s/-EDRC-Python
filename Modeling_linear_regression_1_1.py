#선형 회귀 분석

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

#Data
#Excel data 읽어오기
data_df = pd.read_excel("excel_linear_data_1.xlsx")
print(data_df)

#Array data로 바꾸기
data = data_df.values
print(data)

#x_data
x_data = data_df.iloc[:,0]
x_data = x_data.values

#y_data
y_data = data_df.iloc[:,1]
y_data = y_data.values

print(x_data.shape)
print(y_data.shape)

plt.figure(1)
plt.scatter(data[:, 0], data[:, 1])
plt.title("Data")
plt.xlabel("x")
plt.ylabel("y")
# plt.axis([0, 420, 0, 50])
plt.show()

#데이터 분리
x_data = data[:,0].reshape(-1,1) #row data 형태로 만들어 줌.
y_data = data[:,1].reshape(-1,1) #row data 형태로 만들어 줌.

#임의의 파라미터
y_predict1 = -10*x_data-1
y_predict2 = -5*x_data-2

plt.figure(2)
plt.scatter(x_data, y_data, label='data')
plt.plot(x_data,y_predict1, label='predict_1')
plt.plot(x_data,y_predict2, label='predict_2')
plt.legend(loc='upper left')
plt.title("Linear model, Different parameters")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# print(x_data)
x_data_mod = np.insert(x_data, 0, 1, axis=1) #상수 값 부분을 x 축 벡터에 합쳐서 표현
# print(x_data_mod)
# print(x_data_mod.T)

#Analytical solution
coeff = np.linalg.inv(
	x_data_mod.T.dot(x_data_mod)).dot(x_data_mod.T).dot(y_data)
print(coeff)
coeff_a = coeff[1,0]
coeff_b = coeff[0,0]
print("Analytical solution (slope): \n", coeff_a)
print("Analytical solution (bias): \n", coeff_b)
y_predict_analytical = coeff_a*x_data + coeff_b

#Figure
plt.figure(3)
plt.scatter(x_data, y_data, label='data')
plt.plot(x_data, y_predict_analytical, 'r--', label='predict_analytical')
plt.legend(loc='upper left')
plt.title("Linear model, Analytical solution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Error cost
def Linear_regression_cost(x_data_mod, y_data, p):
	y_predict = np.matmul(x_data_mod,p)
	N = np.size(x_data_mod,0)
	cost = sum((y_data-y_predict)**2)/N
	return cost

cost = Linear_regression_cost(x_data_mod, y_data, [[1],[0.1]])
print("Random selection cost: ", cost)
cost_analytical = Linear_regression_cost(x_data_mod, y_data, [[coeff_b],[coeff_a]])
print("Analtical solution cost: ", cost_analytical)

def Calculate_new_parameter(x_data_mod, y_data, p):
	p=np.array(p)
	step_size=0.001
	N = np.size(x_data_mod, 0)
	Gradient = np.matmul((np.matmul(x_data_mod, p)-y_data).T, x_data_mod)/N
	p_new = p.T-step_size*Gradient
	p_new = p_new.T
	return p_new

# print(Calculate_new_parameter(x_data_mod, y_data, [[11], [0.1]]))
#inital guess
p_new = np.array([[1.],[0.1]])
cost = Linear_regression_cost(x_data_mod, y_data, p_new)

for i in range(101):
	p_new_new = Calculate_new_parameter(x_data_mod, y_data, p_new[:, i].reshape(-1,1))
	p_new = np.hstack((p_new, p_new_new))
	# print(p_new_new)
	cost_new = Linear_regression_cost(x_data_mod, y_data, p_new_new)
	cost = np.vstack((cost, cost_new))
	print(cost_new)

coeff_a_numerical = p_new[0,100]
coeff_b_numerical = p_new[1,100]
print("Numerical solution (slope): \n", coeff_b_numerical)
print("Numerical solution (bias): \n", coeff_a_numerical)
y_predict_numerical = coeff_b_numerical*x_data + coeff_a_numerical

# Figure
plt.figure(4)
plt.subplot(2,1,1)
plt.scatter(x_data, y_data, label='data')
plt.plot(x_data, y_predict_numerical, 'r-', label='predict_numerical')
plt.legend(loc='upper left')
plt.title("Linear model, Numerical solution")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(2,1,2)
plt.plot(cost, 'r-', label='predict_numerical')
plt.title("Error cost")
plt.xlabel("Run")
plt.ylabel("Error cost")
plt.show()