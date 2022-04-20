#tensorflow 기본 개념 3

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

#변수 설정 (예를 들면 placeholder로 받은 데이터로 찾아야하는 파라미터 값)
var1 = tf.Variable([3],dtype=tf.float32)
var2 = data1*var1
sess2 = tf.Session()
init1 = tf.global_variables_initializer() #(*)필수
sess2.run(init1) #(*)필수
result6 = sess2.run(var2, feed_dict={data1: input_data})
print("Variable 예제의 결과: ", result6)

#변수 행렬 연산
var_mat1 = tf.Variable([[10, 20]], dtype = tf.float32)
var_mat2 = tf.matmul(var_mat1, con_mat2)
sess3 = tf.Session()
init2 = tf.global_variables_initializer()
sess3.run(init2)
print("Variable 행렬 연산의 결과: ", sess3.run(var_mat2))