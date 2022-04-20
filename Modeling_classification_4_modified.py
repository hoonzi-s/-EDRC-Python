#Softmax 이용해서 3개 이상 분류하기

#Tensorflow 이용하여 데이터 구분하기

import tensorflow as tf
import numpy as np
import pandas as pd
import openpyxl

#Excel data 읽어오기
data_df = pd.read_excel("excel_softmax_data_1.xlsx")
print(data_df)

#Array data로 바꾸기
data = data_df.values
print(data)

#y를 one hot encoding으로 바꾸기
data_onehot = pd.get_dummies(data_df)
print(data_onehot)

#x_data: (10 by 4)
x_data = data_onehot.iloc[:,0:5]
x_data = x_data.values

#y_data: (10 by 3)
y_data = data_onehot.iloc[:,5:]
y_data = y_data.values

print(x_data.shape)
print(y_data.shape)

#한 줄로 표현
# model = tf.keras.Sequential([tf.keras.layers.Dense(units=3, activation='softmax', input_shape=x_data[0].shape) ]) #input dimension 5, output dimension 3

#여러 줄로 쪼개서 표현
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=3, activation='softmax', input_shape=x_data[0].shape))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=100)