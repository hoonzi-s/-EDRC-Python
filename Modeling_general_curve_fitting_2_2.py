#일반적인 curve fitting 예제 (scipy.optimization 패키지 이용)
#Nonlinear + multivariable

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Data
data = np.array([[100, 30, 20],
		        [100, 32, 21],
		        [110, 35, 21],
		        [120, 37, 25],
		        [130, 37, 18],
		        [130, 40, 22],
		        [150, 41, 24],
		        [190, 45, 30],
		        [200, 47, 35],
		        [240, 49, 32],
		        [255, 50, 30],
		        [270, 55, 29],
		        [300, 55, 25],
		        [350, 57, 23],
		        [400, 60, 19]])

#데이터 분리
x1_data = data[:,0] #x_data = data[:,0].reshape(-1,1)으로 하면 row vector로 되어서 ㄴㄴ. column vector로 해야함.
x2_data = data[:,1]
y_data = data[:,2]
# print(type(x_data))
# print(type(data[:,0].reshape(-1,1)))

#모델 형태 지정 2
def Nonlinear_model(X,coeff1, coeff2, coeff3, coeff4, coeff5, coeff6):
    x1, x2 = X
    return coeff1*x1**2 + coeff2*x2**2 + coeff3*x1*x2 + coeff4*x1 + coeff5*x2 + coeff6

#Data fitting 2
popt2, pcov2 = curve_fit(Nonlinear_model, (x1_data,x2_data), y_data)
print(popt2) #fitting parameters
y_predict_nonlinear = Nonlinear_model((x1_data, x2_data), popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5])

#그래프 그리기
x1 = np.arange(100, 400)
x2 = np.arange(30, 60)
x1_grid, x2_grid = np.meshgrid(x1,x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid,
                Nonlinear_model((x1_grid, x2_grid),popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5]),
                cmap='Wistia', label="Model")
ax.scatter(x1_data, x2_data, y_data, marker='o', c='r',label='Data')
# plt.legend(loc='best')
plt.show()