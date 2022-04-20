#numpy 패키지 예제
#빈 껍데기 (x) 안에 data 값 받아서 연산한 결과 넣고 출력하기

import numpy as np

data = np.array([1,2,3,4,5])

x=[] #빈 껍데기

for i in range(5):
    new_x = data[i]**2
    x=np.append(x,new_x)

print('data 값은: \n',data)
print('x 값은: \n', x)