#A type과 B type 분류하기 (Tensorflow)

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

data = np.array([[1, 0],
		        [2, 0],
		        [3, 0],
		        [4, 0],
		        [5, 0],
		        [6, 1],
		        [7, 1],
		        [8, 1],
		        [9, 1],
		        [10, 1]])

#데이터 분리
x_data = data[:,0].reshape(-1,1) #row data 형태로 만들어 줌.
y_data = data[:,1].reshape(-1,1) #row data 형태로 만들어 줌.

x_data_mod = np.insert(x_data, 0, 1, axis=1) #상수 값 부분을 x 축 벡터에 합쳐서 표현

#placeholder
x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

#parameters
p = tf.Variable(tf.random_normal((2,1)))

#model
y_model = tf.sigmoid(tf.matmul(x,p))

#Error cost
cost = -tf.reduce_mean(y * tf.log(y_model) + (1 - y) * (tf.log(1 - y_model)))

#학습
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#세션 생성
sess1 = tf.Session()

#세션 초기화 #(*)필수
init1 = tf.global_variables_initializer()
sess1.run(init1)

#학습 (최적 파라미터 찾기)
for i in range(20001):
    sess1.run(train, feed_dict={x: x_data_mod, y: y_data})

    if i%1000 == 0:
        print("Run 횟수: ",i, "Error cost: ", sess1.run(cost, feed_dict={x: x_data_mod, y: y_data}))

print("x 값 넣었을 때의 y 예측 값: ",sess1.run(y_model, feed_dict={x: [[1,6]]}))