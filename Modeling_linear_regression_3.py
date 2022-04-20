#선형 회귀 분석 (tensorflow 이용해서)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#데이터 바꿔야 함. Excel file에서 pandas로 읽어오도록 해보자.
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

#Parameter
W = tf.Variable([1.0], dtype=tf.float32) #혹은 범위로 줘도 상관없음.
b = tf.Variable([0.0], dtype=tf.float32)

#Model
model = W*x_data + b

#Error cost
cost = tf.reduce_mean(tf.square(model - y_data))

rate = tf.Variable(0.002)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

#session 생성
sess = tf.Session()

#session 초기화 (*) 필수
init = tf.global_variables_initializer()
sess.run(init)

Error_cost=np.array([0])

for i in range(101):
    sess.run(train)
    Error_cost=np.hstack((Error_cost, sess.run(cost)))

    if i%10 == 0:
        print(i, sess.run(cost), sess.run(W), sess.run(b))

#최종 결과물
coeff = sess.run(W)
bias  = sess.run(b)
y_predict = coeff*x_data + bias

#Session 종료
sess.close()

#시각화
plt.figure(1)
plt.subplot(1,2,1)
plt.title('Linear model, tensorflow')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_data, y_data, label='data')
plt.plot(x_data, y_predict, 'r-', label='predict_tensorflow')
plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.title('Cost')
plt.plot(Error_cost[1:])
plt.xlabel("Run")
plt.ylabel("Error cost")
plt.show()


