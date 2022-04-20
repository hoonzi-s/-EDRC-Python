#Scipy 패키지 예제 (optimization이나 미분, 적분에 많이 사용 됨)

import scipy as sp
import scipy.integrate as integrate

#적분 할 함수
def fun(x):
    return x**2

result1 = integrate.quad(fun,0,1)
print("적분한 값은: \n", result1)
print("적분한 값 중 진짜는: \n", result1[0])
