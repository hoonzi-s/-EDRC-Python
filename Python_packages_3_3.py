#Pandas 패키지 예제

import pandas as pd
import numpy as np

#Series 자료 구조
e = pd.Series([-1, 1, 10, 5])
print(e)

#index 추가 하기
e.index = ['index1','index2', 'index3', 'index4']

#index 애초에 넣기
e = pd.Series([-10, 5, 4, 11], index=['index1', 'index2', 'index3', 'index4'])