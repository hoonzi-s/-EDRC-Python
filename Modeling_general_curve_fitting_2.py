#일반적인 curve fitting 예제 (scipy.optimization 패키지 이용)
#Nonlinear

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Data
data = np.array([[100, 20],
		[100, 21],
		[110, 21],
		[120, 25],
		[130, 18],
		[130, 22],
		[150, 24],
		[190, 30],
		[200, 35],
		[240, 32],
		[255, 30],
		[270, 29],
		[300, 25],
		[350, 23],
		[400, 19],])

#데이터 분리
x_data = data[:,0] #x_data = data[:,0].reshape(-1,1)으로 하면 row vector로 되어서 ㄴㄴ. column vector로 해야함.
y_data = data[:,1]
print(type(x_data))
print(type(data[:,0].reshape(-1,1)))

#모델 형태 지정 2
def Nonlinear_model(x, coeff1, coeff2, bias):
    return coeff2*x**2 + coeff1*x_data + bias

#Data fitting 2
popt2, pcov2 = curve_fit(Nonlinear_model, x_data, y_data)
print(popt2) #fitting parameters
print(pcov2)
y_predict_nonlinear = Nonlinear_model(x_data, popt2[0], popt2[1], popt2[2])
plt.figure(2)
plt.scatter(data[:, 0], data[:, 1],label='Data')
plt.plot(x_data, y_predict_nonlinear,'r--',label='Nonlinear model')
plt.legend(loc='best')
plt.title("Nonlinear curve fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
