#앞의 state가 뒤의 state에 영향을 주는 시스템
#AR + I + MA = ARIMA model
#AR: Autoregression
#I:  Integrated
#MA: Moving average
#데이터 수 적을 때
#상수 항 붙였을 때

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt

#Data 바꾸긴 해야함 (pandas로 excel 파일 읽어오자!)
data = np.array([1, 3, 6, 22, 20, 19, 15, 14, 13, 9])

#시각화
# plt.figure(1)
# plt.plot(data)
# plt.show()

# #하이퍼 파라미터 찾기
# plot_acf(data)
# plot_pacf(data)
# plt.show()
#
# #차분 계수 찾기
# data_diff_1=[]
# for i in range(len(data)-1):
#     new_data_diff_1=data[i+1]-data[i]
#     data_diff_1=np.append(data_diff_1,new_data_diff_1)
# print(data_diff_1)
# plot_acf(data_diff_1)
# plot_pacf(data_diff_1)
# plt.show()

model = ARIMA(data, order=(1,1,0))
model_fit = model.fit(trend='c', full_output='True', disp=1)
print(model_fit.summary())
model_fit.plot_predict()
plt.show()