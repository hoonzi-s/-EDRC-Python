#Softmax 이용해서 3개 이상 분류하기

#Tensorflow 이용하여 데이터 구분하기

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

print(x_data.dtype)
print(y_data.dtype)

#여러 줄로 쪼개서 표현
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=3, activation='softmax', input_shape=x_data[0].shape))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=100)