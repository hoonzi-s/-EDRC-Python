#Pandas 패키지 예제 (대량의 데이터를 읽어오고 처리하는 곳에 사용됨)

import pandas as pd
import numpy as np

#DataFrame 자료 구조

#Data 1: column matrix
a = pd.DataFrame([[1, 3, 5, np.nan, 9]])

#Print data 1
print("matrix a는:\n", a)
print("matrix a의 사이즈는:\n", a.shape)
print("a에서 첫번째 ([0]) 행(row)의 값은:\n", a.iloc[0,:])
print("a에서 두번째 ([1]) 행(row)의 값은:\n", a.iloc[0,0])

#Data 2: row matrix
b = pd.DataFrame([[1],[3],[5],[7],[9]])

#Print data 2
print("matrix b는: ", b)
print("b에서 세번째 ([2]) 행(row)의 값은:\n", b.iloc[2])