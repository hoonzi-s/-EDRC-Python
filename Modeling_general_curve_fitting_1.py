#일반적인 curve fitting 예제 (scipy.optimization 패키지 이용)
#Linear

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Data
data = np.array([[100, 20],
		[150, 24],
		[170, 27],
		[200, 30],
		[250, 36],
		[260, 38],
		[290, 40],
		[400, 55],
		[410, 57],
		[430, 60],
		[440, 61],
		[460, 62],
		[500, 68],
		[550, 72],
		[600, 80]])

#데이터 분리
x_data = data[:,0] #x_data = data[:,0].reshape(-1,1)으로 하면 row vector로 되어서 ㄴㄴ. column vector로 해야함.
y_data = data[:,1]
print(type(x_data))
print(type(data[:,0].reshape(-1,1)))

#모델 형태 지정 1
def Linear_model(x_data, coeff, bias):
    return coeff*x_data + bias

#Data fitting 1
popt, pcov = curve_fit(Linear_model, x_data, y_data)
print(popt)
print(pcov)
y_predict_linear = Linear_model(x_data, popt[0], popt[1])
plt.figure(1)
plt.scatter(data[:, 0], data[:, 1],label='Data')
plt.plot(x_data, y_predict_linear,'r--',label='Linear model')
plt.title("Linear curve fitting")
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

