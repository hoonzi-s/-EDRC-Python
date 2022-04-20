#numpy 패키지 예제

import numpy as np

a = [2,4,6,7,10] #list
b = np.array(a)  #numpy 배열 생성
print("list a는: \n", a)
print("numpy 배열 b는: \n", b)
print("numpy 배열 b의 차원은: \n", b.shape)
print("numpy 배열 b의 자료형은: \n", b.dtype)

print("---------------------------------------------------------------------------------------------------------------")
a_ = [[2,4,6,7,10]] #row matrix
b_ = np.array(a_)
print("row matrix a_는: \n", a_)
print("numpy matrix b_는: \n", b_)
print("numpy 배열 b_의 차원은: \n", b_.shape)
print("numpy 배열 b_의 자료형은: \n", b_.dtype)

print("---------------------------------------------------------------------------------------------------------------")
c=np.array([[1],[3],[5],[7],[9]]) #column matrix
print("numpy 배열 c는: \n", c)
print("numpy 배열 c의 차원은: \n", c.shape)

print("---------------------------------------------------------------------------------------------------------------")
d=np.array([[1,2,3],[4,5,6]]) #general matrix
print("numpy 배열 d는: \n", d)
print("numpy 배열 d의 차원은: \n", d.shape)
print("numpy 배열 d의 1행 2열 값은: \n", d[0,1])
print("numpy 배열 d의 2행 3열 값은: \n", d[1,2])
print("numpy 배열 d의 1열 모든 행의 값은: \n", d[:,0])
print("numpy 배열 d의 2행 모든 열의 값은: \n", d[1,:])
print("numpy 배열 d의 행끼리의 합은: \n", np.sum(d, axis=0))
print("numpy 배열 d의 열끼리의 합은: \n", np.sum(d, axis=1))
print("numpy 배열 d의 모든 요소의 합은: \n", np.sum(d))

print("---------------------------------------------------------------------------------------------------------------")
e = np.array([[1,3],[5,7]])
f = np.array([[2,4],[6,8]])
print("행렬 e와 f를 element끼리 곱하면: \n", e*f)
print("numpy 배열 e와 f의 행렬 곱 (product)는: \n", np.dot(e,f))
print("e와 f를 np.matmul을 사용하여 곱하면: \n", np.matmul(e,f))

print("---------------------------------------------------------------------------------------------------------------")
print(np.zeros(5))
print(np.ones(5))
print(np.linspace(1,3,num=5))
print(np.random.randn(2,3))
print(np.abs(-2))
print(np.sqrt(64))
print(np.square(3))
print(np.exp(2))
print(np.log(2))
print(np.log10(1000))

