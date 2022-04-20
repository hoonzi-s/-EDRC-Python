#선형 회귀 분석 (tensorflow 이용해서, multi-dimension)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#Data
x1_data = [ 2.,   1.,  3.,  0., 7.]
x2_data = [ 0.,   2.,  3.,  4., 5.]
y_data  = [8.3, -3.5,   0, -15, 6.]

#Parameters
W1 = tf.Variable([3], dtype=tf.float32)
W2 = tf.Variable([4],  dtype=tf.float32)
b  = tf.Variable([1.],  dtype=tf.float32)

#Model
hypothesis = W1*x1_data + W2*x2_data + b

#Error cost
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

rate = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

#Session 생성 및 초기화
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

Error_cost=[]

for step in range(101):

    sess.run(train)
    Error_cost=np.append(Error_cost,sess.run(cost))

    if step%10 == 0:
        print(step, sess.run(cost), sess.run(W1),
              sess.run(W2), sess.run(b))

W1_final = sess.run(W1)
W2_final = sess.run(W2)
b_final  = sess.run(b)

sess.close()

#정리
x1              = np.linspace(0,10)
x2              = np.linspace(0,10)
x1_mesh,x2_mesh = np.meshgrid(x1,x2)
y_predict       = W1_final*x1_mesh + W2_final*x2_mesh + b_final

#시각화
fig=plt.figure(1)
ax1=fig.add_subplot(121,projection='3d')
ax1.scatter(x1_data, x2_data, y_data, label='data', marker='o', c='r')
ax1.plot_surface(x1_mesh,x2_mesh,y_predict, label='model')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
plt.title('Data fitting result using tensorflow')
# plt.legend()

ax2=fig.add_subplot(122)
ax2.plot(Error_cost)
plt.title('Error cost')
plt.xlabel('Training number')
plt.ylabel('Error cost')
plt.show()