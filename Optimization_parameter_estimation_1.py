#일반적인 Nonlinear static 함수의 parameter fitting

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#예제
#예제는 데이터를 pandas로 읽어 오는 거로 바꾸자. 좀 더 general 하게
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

#모델 형태 지정 2
def Nonlinear_model(x_data, coeff1, coeff2, bias):
    return coeff2*x_data**2 + coeff1*x_data + bias

#Data fitting 2
popt2, pcov2 = curve_fit(Nonlinear_model, x_data, y_data)
print(popt2)
print(pcov2)
y_predict_nonlinear = Nonlinear_model(x_data, popt2[0], popt2[1], popt2[2])
plt.figure(2)
plt.scatter(data[:, 0], data[:, 1],label='Data')
plt.plot(x_data, y_predict_nonlinear,'r-',label='Nonlinear model')
plt.title("Data fitting example")
plt.xlabel("x")
plt.ylabel("y")
plt.show()