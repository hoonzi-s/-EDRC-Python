#Partial least squares 기초 (Data reduction)

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

#data (바꿀 예정. pandas로 엑셀 파일 읽어오자.)
X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

#데이터 표준화 (Normalize, 평균 =0, 표준편차 1)
mean_X=np.mean(X,axis=0)
std_X=np.std(X,axis=0)
X_ = StandardScaler().fit_transform(X) #X_=(X-mean(X))/std(X)
# X_re = X_*std_X+mean_X (복원)

mean_Y=np.mean(Y,axis=0)
std_Y=np.std(Y,axis=0)
Y_ = StandardScaler().fit_transform(Y) #X_=(X-mean(X))/std(X)

#PLS
pls2 = PLSRegression(n_components=2) #객체 생성
pls2.fit(X_, Y_) #PLS로 loading, score, weight and coefficient matrix (+ normalize)
print("X block loading matrix: (normalized)", pls2.x_loadings_)
print("X block score matrix: (normalized)", pls2.x_scores_)
print("X block weights matrix: (normalized)", pls2.x_weights_)
print("---------------------------------------------------------------------------------------------------------------")
print("Y block loading matrix: (normalized)", pls2.y_loadings_)
print("Y block score matrix: (normalized)", pls2.y_scores_)
print("Y block weights matrix: (normalized)", pls2.y_weights_)
print("---------------------------------------------------------------------------------------------------------------")
print("The coefficient matrix (Y_pls = X*coeff): (normalized) \n", pls2.coef_)

Y_pred_ = pls2.predict(X_)
Y_pred  = Y_pred_*std_Y + mean_Y
print("PLS를 통해 차원 줄였다가 복원한 데이터로 재예측한 Y: \n", Y_pred)
# print("Predicted Y value by PLS: ", Y_pred)
# X_score, Y_score = pls2.fit_transform(X, Y)
# print("X score by PLS: ", X_score)
# print("Y score by PLS: ", Y_score)