#A type과 B type 분류하기

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = np.array([[1, 1, 0],
		         [2, 2, 0],
		         [3, 3, 0],
		         [4, 4, 0],
		         [5, 5, 1],
		         [6, 6, 1],
		         [7, 7, 1],
		         [8, 8, 1],
		         [9, 9, 1],
		         [10, 10, 1]])

#데이터 분리
x_data = data[:,0:1].reshape(-1,1) #row data 형태로 만들어 줌.
y_data = data[:,2]

#model 생성
model = LogisticRegression(random_state=0, solver='lbfgs')
solution = model.fit(x_data,y_data)
print(solution.predict(x_data))
print(solution.score(x_data,y_data))