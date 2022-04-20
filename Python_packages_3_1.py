#Pandas 패키지 예제

import pandas as pd
import numpy as np

#DataFrame 자료 구조

#Data 3: general matrix
c = pd.DataFrame(np.random.randn(3,2))

#Print data 3
print("matrix c는:\n", c)

print("c에서 첫번째 ([0]) 행(row)의 값은:\n", c.iloc[0])
print("c에서 첫번째 ([0]) 열(column)의 값은: \n", c.iloc[:,0])

#Data 4: general matrix + 각 행과 열에 이름 추가
d = pd.DataFrame(np.random.randn(3,2), index=['row1','row2','row3'], columns=['col1','col2'])

#Print data 3
print("matrix d는:\n", d)

print("d에서 row1 행(row)의 값은:\n", d.loc['row1',:])
print("c에서 col2 열(column)의 값은: \n", d.loc[:,'col2'])