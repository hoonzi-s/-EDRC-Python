#Scipy 패키지 예제 (optimization이나 미분, 적분에 많이 사용 됨)

import scipy as sp
import scipy.integrate as integrate
import numpy as np

#Ax=b의 솔루션
A=sp.array([[10,5,3.5],[5,0,0.5],[2,-1,2]])
b=sp.array([2.5,0,10])
x_sp = sp.linalg.solve(A,b)
A_eig_val, A_eig_vec = sp.linalg.eig(A)
print("Scipy로 구한 솔루션은: ", x_sp)
print("A의 역행렬은: \n", sp.linalg.inv(A))
print("A의 eigen value는: \n", A_eig_val)
print("A의 eigen vector는: \n", A_eig_vec)


x_np = np.linalg.solve(A,b)
print("Numpy로 구한 솔루션은: ", x_np)