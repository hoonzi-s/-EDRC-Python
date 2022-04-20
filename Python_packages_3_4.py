#Pandas 패키지 예제
#excel file에 저장하기

import pandas as pd
import numpy as np

#데이터 생성
x=np.linspace(0,10,num=100).reshape(-1,1)
y1=-np.sin(x).reshape(-1,1)
y2=np.cos(x)**2
y2=y2.reshape(-1,1)

#데이터 합치기
data=np.hstack((x,y1))
data=np.hstack((data,y2))

#pandas의 DataFrame 형식에 넣기
e = pd.DataFrame(data)
print(e)

#xlsx 파일로 저장하기
e.to_excel('excel_pandas_example.xlsx')

