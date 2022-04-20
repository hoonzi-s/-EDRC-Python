import tensorflow as tf
import numpy as np
from tensorflow import keras

# x_data = [11,12,13,14,15,16,17]
# y_data = [148, 154, 170, 165, 175, 183, 190]

data = np.array([[1, 3],
				[2, 3.2],
				[2.3, 4],
				[3.2, 5],
				[3.3, 5.3],
				[4, 6],
				[4.5, 6.8],
				[4.8, 7.1],
				[4.9, 7.2],
				[6.2, 9.3],
				[7, 10.1],
				[7.1, 10.9],
				[8.8, 14.1],
				[9.5, 16.0],
				[10, 17]])

#Data
x_data = data[:,0]
y_data = data[:,1]

learning_rate = 0.001

#tf.keras.layers.Dense에서
#units: 해당 층의 output 개수 (즉 마지막 층에서는 y_data dimension과 동일해야만 한다.)
#activation: nonlinearity를 설명해주는 활성화 함수의 종류
#input_shape: 첫 번째 층에서만 넣어주면 되고, x_data dimension과 동일해야만 한다.)

#모델 생성: 여러 줄로 표현
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1, activation='linear', input_shape=(1,)))

#컴파일 준비
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')
# sgd = tf.keras.optimizers.SGD(learning_rate = 0.001)
# model.compile(loss='mean_squared_error',optimizer=sgd)

#Fitting
model.fit(x_data, y_data, epochs=100)

model.evaluate(x_data, y_data)

print(f'x가 4일 때, y의 예측값 : {model.predict(np.array([4]))[0][0]}')