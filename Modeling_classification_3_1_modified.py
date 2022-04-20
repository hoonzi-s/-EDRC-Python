#Softmax 이용해서 3개 이상 분류하기

#Tensorflow 이용하여 데이터 구분하기

#Hidden layer 추가

import tensorflow as tf
import numpy as np

#x_data: (10 by 4)
x_data = np.array([[0., 0., 1., 1.],
                   [1., 1., 0., 1.],
                   [1., 0., 1., 1.],
                   [0., 1., 0., 1.],
                   [1., 0., 1., 0.],
                   [1., 0., 0., 1.],
                   [1., 1., 1., 0.],
                   [0., 0., 1., 0.],
                   [1., 0., 0., 1.],
                   [0., 0., 0., 1.]], dtype='f')

#y_data: (10 by 3)
y_data = np.array([[1.,0.,0.],
                   [0.,1.,0.],
                   [1.,0.,0.],
                   [0.,0.,1.],
                   [1.,0.,0.],
                   [0.,1.,0.],
                   [0.,0.,1.],
                   [1.,0.,0.],
                   [0.,0.,1.],
                   [0.,0.,1.]], dtype='f')

print(x_data.shape)
print(y_data.shape)

#input dimension 4, output dimension 3
#tf.keras.layers.Dense에서
#units: 해당 층의 output 개수 (즉 마지막 층에서는 y_data dimension과 동일해야만 한다.)
#activation: nonlinearity를 설명해주는 활성화 함수의 종류
#input_shape: 첫 번째 층에서만 넣어주면 되고, x_data dimension과 동일해야만 한다.)

#모델 생성: 한 줄로 표현
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=10, activation='relu', input_shape=x_data[0].shape),
#     tf.keras.layers.Dense(units=24, activation='relu'),
#     tf.keras.layers.Dense(units=12, activation='relu'),
#     tf.keras.layers.Dense(units=3, activation='softmax') ])

#모델 생성: 여러 줄로 표현
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_shape=x_data[0].shape))
model.add(tf.keras.layers.Dense(units=24, activation='relu'))
model.add(tf.keras.layers.Dense(units=12, activation='relu'))
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

#컴파일 준비
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting
model.fit(x_data, y_data, epochs=100)

# model.summary()