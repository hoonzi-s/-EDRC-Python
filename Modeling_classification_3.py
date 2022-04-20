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

#Model
x = tf.placeholder(tf.float32, [None, 4], name="input")
y = tf.placeholder(tf.float32, [None, 3], name='output')
W = tf.Variable(tf.random_normal([4,3]))
b = tf.Variable(tf.random_normal([3]))
L = tf.add(tf.matmul(x, W),b)
y_predict = tf.nn.softmax(L)

#Objective function (softmax이므로 cross-entropy cost function 쓰자.)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=L))

#학습 빠르기
rate = 0.9

#파라미터 찾기
train = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(cost)

#세션 생성
sess1 = tf.Session()

#세션 초기화 #(*)필수
init1 = tf.global_variables_initializer()
sess1.run(init1)

#학습 (최적 파라미터 찾기)
for i in range(10001):
    sess1.run(train, feed_dict={x: x_data, y: y_data})

    if i%1000 == 0:
        print("Run 횟수: ",i, "Error cost: ", sess1.run(cost, feed_dict={x: x_data, y: y_data}))

print("첫 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(y_predict, feed_dict={x: [[0., 0., 1., 1.]]}))
print("두 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(y_predict, feed_dict={x: [[1., 1., 0., 1.]]}))
print("세 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(y_predict, feed_dict={x: [[1., 0., 1., 1.]]}))
print("네 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(y_predict, feed_dict={x: [[0., 1., 0., 1.]]}))
print("다섯 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(y_predict, feed_dict={x: [[1., 0., 1., 0.]]}))

# print("첫 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(tf.argmax(sess1.run(y_predict, feed_dict={x: [[0., 0., 1., 1.]]}),1)))
# print("두 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(tf.argmax(sess1.run(y_predict, feed_dict={x: [[1., 1., 0., 1.]]}),1)))
# print("세 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(tf.argmax(sess1.run(y_predict, feed_dict={x: [[1., 0., 1., 1.]]}),1)))
# print("네 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(tf.argmax(sess1.run(y_predict, feed_dict={x: [[0., 1., 0., 1.]]}),1)))
# print("다섯 번 째 x 값 넣었을 때의 y 예측 값: ",sess1.run(tf.argmax(sess1.run(y_predict, feed_dict={x: [[1., 0., 1., 0.]]}),1)))

sess1.close()