#데이터 시각화 기초

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl

#Excel data 읽어오기
data_df = pd.read_excel("excel_linear_data_1.xlsx")
print(data_df)

#Array data로 바꾸기
data = data_df.values
print(data)

#y를 one hot encoding으로 바꾸기
data_onehot = pd.get_dummies(data_df)
print(data_onehot)

#x_data
x_data = data_onehot.iloc[:,0]
x_data = x_data.values

#y_data
y_data = data_onehot.iloc[:,1]
y_data = y_data.values

print(x_data.shape)
print(y_data.shape)

plt.figure(1)
plt.plot(x_data,y_data,'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Figure of data from pandas')
plt.show()