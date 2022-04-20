#일반적인 curve fitting 예제 (scipy.optimization 패키지 이용)
#Linear

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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