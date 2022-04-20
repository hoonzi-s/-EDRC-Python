#일반적인 curve fitting 예제 (scipy.optimization 패키지 이용)
#Nonlinear + multivariable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Excel data 읽어오기
data_df = pd.read_excel("excel_nonlinear_data_2.xlsx")
print(data_df)

#Array data로 바꾸기
data = data_df.values
print(data)

#x_data
x_data = data_df.iloc[:,0:3]
x_data = x_data.values

#y_data
y_data = data_df.iloc[:,3]
y_data = y_data.values

print(x_data.shape)
print(y_data.shape)

#데이터 분리
x1_data = x_data[:,0] #x_data = data[:,0].reshape(-1,1)으로 하면 row vector로 되어서 ㄴㄴ. column vector로 해야함.
x2_data = x_data[:,1]
x3_data = x_data[:,2]
# print(type(x_data))
# print(type(data[:,0].reshape(-1,1)))

#모델 형태 지정 2
def Nonlinear_model(X,coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7):
    x1, x2, x3 = X
    return coeff1*x1**2 + coeff2*x2**2 + coeff3*x3**2 + coeff4*x1*x2 + coeff5*x1 + coeff6*x2 + coeff7

#Data fitting 2
popt2, pcov2 = curve_fit(Nonlinear_model, (x1_data,x2_data,x3_data), y_data)
print(popt2) #fitting parameters
y_predict_nonlinear = Nonlinear_model((x1_data, x2_data, x3_data), popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5], popt2[6])

#그래프 그리기
y_pred=Nonlinear_model((x1_data, x2_data, x3_data),popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5], popt2[6])
print(y_pred)

plt.figure(1)
plt.plot(y_data,'bo',label='y_data')
plt.plot(y_pred,'ro',label='y_pred')
plt.xlabel('index')
plt.ylabel('Output')
plt.legend(loc='best')
plt.title('Data & prediction')
plt.show()