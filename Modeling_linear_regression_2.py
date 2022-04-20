#선형 회귀 분석 (scikit-learn 이용해서)

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = np.array([[1, 3],
				[2, 3.2],
				[2.3, 4],
				[3.2, 5],
				[3.3, 5.3],
				[4, 6],
				[4.5, 6.8],
				[4.8, 7.1],
				[4.9, 7.2],
				[6.2, 9.3],
				[7, 10.1],
				[7.1, 10.9],
				[8.8, 14.1],
				[9.5, 16.0],
				[10, 17]])

#데이터 분리
x_data = data[:,0].reshape(-1,1) #reshape?
y_data = data[:,1].reshape(-1,1)

#선형 회귀 분석 (y_model = b0+b1*x)
model = LinearRegression()
model.fit(x_data, y_data)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_model = model.intercept_ + model.coef_*x_data

#그래프
plt.figure(1)
plt.plot(x_data,y_data, 'bo', label='y_data')
plt.plot(x_data,y_model, 'r-', label='y_model')
plt.legend(loc='upper left')
plt.title('Linear model, scikit-learn')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

