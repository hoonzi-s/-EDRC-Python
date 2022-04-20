#선형 회귀 분석 (tensorflow 이용해서, multi-dimension)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#Data
x_data  = np.array([[2.,   1.,  3.,  0., 7.],
                    [0.,   2.,  3.,  4., 5.]])
x_data  = np.transpose(x_data)
y_data  = [8.3, -3.5,   0, -15, 6.]
y_data  = np.transpose(y_data)

learning_rate = 0.001

#tf.keras.layers.Dense에서
#units: 해당 층의 output 개수 (즉 마지막 층에서는 y_data dimension과 동일해야만 한다.)
#activation: nonlinearity를 설명해주는 활성화 함수의 종류
#input_shape: 첫 번째 층에서만 넣어주면 되고, x_data dimension과 동일해야만 한다.)

#모델 생성: 여러 줄로 표현
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1, activation='linear', input_shape=x_data[0].shape))
# model.add(tf.keras.Input(shape=(2,1))) # input dimension=2
# model.add(tf.keras.layers.Dense(1, activation='linear')) # output dimension=1
# model.add(tf.keras.layers.Dense(units=3, activation='softmax', input_shape=x_data[0].shape))

#컴파일 준비
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')
# sgd = tf.keras.optimizers.SGD(learning_rate = 0.001)
# model.compile(loss='mean_squared_error',optimizer=sgd)

#Fitting
history = model.fit(x_data, y_data, epochs=100)

plt.figure(1)
plt.plot(history.history['loss'])
plt.ylabel('Loss (mean squared error)')
plt.xlabel('Epochs')
plt.show()