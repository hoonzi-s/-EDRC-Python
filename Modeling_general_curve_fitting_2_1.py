#일반적인 curve fitting 예제 (scipy.optimization 패키지 이용)
#Nonlinear

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Excel data 읽어오기
data_df = pd.read_excel("excel_nonlinear_data_1.xlsx")
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