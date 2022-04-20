#Tensorflow 이용하여 데이터 구분하기

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[1, 0],
		        [2, 0],
		        [3, 0],
		        [4, 0],
		        [5, 1],
		        [6, 1],
		        [7, 1],
		        [8, 1],
		        [9, 1],
		        [10, 1]])

#데이터 분리
x_data = data[:,0].reshape(-1,1) #row data 형태로 만들어 줌.
y_data = data[:,1]

print(x_data.shape)
print(y_data.shape)

#input dimension 1, output dimension 1
#tf.keras.layers.Dense에서
#units: 해당 층의 output 개수 (즉 마지막 층에서는 y_data dimension과 동일해야만 한다.)
#activation: nonlinearity를 설명해주는 활성화 함수의 종류
#input_shape: 첫 번째 층에서만 넣어주면 되고, x_data dimension과 동일해야만 한다.)

#모델 생성: 한 줄로 표현
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=x_data[0].shape)])

#컴파일 준비
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='binary_crossentropy', metrics=['accuracy'])

#Fitting
history=model.fit(x_data, y_data, epochs=100)

plt.figure(1)
plt.plot(history.history['loss'])
plt.ylabel('Loss (binary_crossentropy)')
plt.xlabel('Epochs')
plt.show()

# model.summary()

#prediction
print("x=1일때 sigmoid 예측 값: ", model.predict([1]))
print("x=10일때 sigmoid 예측 값: ", model.predict([10]))