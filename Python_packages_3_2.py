#Pandas 패키지 예제

import pandas as pd
import numpy as np

#DataFrame 자료 구조

#Data 3: general matrix
c = pd.DataFrame(np.random.randn(5,3),index=['row0','row1','row2','row3', 'row4'], columns=['col0','col1', 'col2'])

#Print data 3
print("matrix c는:\n", c)

print("matrix c의 앞에서 2개의 행 값은: \n", c.head(2))
print("matrix c의 뒤에서 2개의 행 값은: \n", c.tail(2))
print("matrix c의 일반적인 정보는: \n", c.describe())