#tensorflow 기본 개념 1

import tensorflow as tf

#상수 지정
con1 = tf.constant(10)
con2 = tf.constant(20, name='con2')

#session 생성
sess1 = tf.Session()

#상수 연산
result1 = tf.add(con1,con2)
print(result1)
print("Tensorflow 덧셈 결과: ", sess1.run(result1))

result2 = tf.subtract(con1,con2)
print("Tensorflow 뺄셈 결과: ", sess1.run(result2))

result3 = tf.multiply(con1,con2)
print("Tensorflow 곱셈 결과: ", sess1.run(result3))

result4 = tf.truediv(con1, con2)
print("Tensorflow 나눗셈 결과: ", sess1.run(result4))

#상수 행렬 연산
con_mat1 = tf.constant([[10, 20]], dtype=tf.float32)
con_mat2 = tf.constant([[30],[40]],dtype=tf.float32)
mat_multiply1 = tf.matmul(con_mat1, con_mat2)
mat_multiply2 = tf.matmul(con_mat2, con_mat1)
print("Tensorflow 행렬 곱_1: \n", sess1.run(mat_multiply1))
print("Tensorflow 행렬 곱_2: \n", sess1.run(mat_multiply2))
