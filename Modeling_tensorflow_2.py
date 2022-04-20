#tensorflow 기본 개념 2

import tensorflow as tf

#상수 지정
con1 = tf.constant(10)
con2 = tf.constant(20, name='con2')

#session 생성
sess1 = tf.Session()

#상수 행렬 연산
con_mat1 = tf.constant([[10, 20]], dtype=tf.float32)
con_mat2 = tf.constant([[30],[40]],dtype=tf.float32)
mat_multiply1 = tf.matmul(con_mat1, con_mat2)
mat_multiply2 = tf.matmul(con_mat2, con_mat1)
print("Tensorflow 행렬 곱_1: ", sess1.run(mat_multiply1))
print("Tensorflow 행렬 곱_2: ", sess1.run(mat_multiply2))

#설정한 값을 넣을 수 있는 그릇: placeholder
input_data=[1,2,3]
data1 = tf.placeholder(dtype=tf.float32)
data2 = data1**2
result5 = sess1.run(data2, feed_dict={data1:input_data})
print("Placeholder 예제의 결과: ", result5)

#Placeholder 행렬 연산
input_data_mat = [[10, 20], [30, 40]]
data_mat1 = tf.placeholder(dtype=tf.float32, shape=[2,2])
data_mat2 = tf.matmul(data_mat1,data_mat1)
print("Placeholder 사용한 행렬 곱: \n", sess1.run(data_mat2,
                                          feed_dict={data_mat1: input_data_mat}))