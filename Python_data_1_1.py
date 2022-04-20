import pandas as pd
import openpyxl

#Excel data 읽어오기
data_df = pd.read_excel("excel_data_1.xlsx")
print('Raw data from xlsx file: \n', data_df)

#Array data로 바꾸기
data = data_df.values
print('Array data로 바꾼 것: \n', data)
print('Array로 바꾼 데이터의 shape: \n', data.shape)

#y를 one hot encoding으로 바꾸기
data_onehot = pd.get_dummies(data_df)
print('One hot encoding으로 바꾼 것: \n', data_onehot)

#x_data
x_data = data_onehot.iloc[:,0:5]
x_data = x_data.values

#y_data
y_data = data_onehot.iloc[:,5:]
y_data = y_data.values

print('x_data의 shape: \n', x_data.shape)
print('y_data의 shape: \n', y_data.shape)