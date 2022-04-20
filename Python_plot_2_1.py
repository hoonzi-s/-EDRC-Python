#데이터 시각화 심화

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import openpyxl

#Excel data 읽어오기
data_df = pd.read_excel("excel_nonlinear_data_2.xlsx")
print(data_df)

#Array data로 바꾸기
data = data_df.values
print(data)

#y를 one hot encoding으로 바꾸기
data_onehot = pd.get_dummies(data_df)
print(data_onehot)

#x_data
x_data = data_onehot.iloc[:,0:3]
x_data = x_data.values

#y_data
y_data = data_onehot.iloc[:,3]
y_data = y_data.values

print(x_data.shape)
print(y_data.shape)

#3D print 기본
fig = plt.figure(1)
ax1  = fig.add_subplot(121,projection='3d')
ax1.plot(x_data[:,0],x_data[:,1],x_data[:,2],'bo')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('x3')
plt.title('3D graph example (x)')

ax2 = fig.add_subplot(122)
ax2.plot(y_data,'ro')
ax2.set_xlabel('index')
ax2.set_ylabel('y')
plt.title('y graph')
plt.grid()
plt.show()