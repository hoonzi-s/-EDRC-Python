#Partial least squares 기초 (Data reduction)

from sklearn.cross_decomposition import PLSRegression

#data (바꿀 예정. pandas로 엑셀 파일 읽어오자.)
X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

#PLS
pls2 = PLSRegression(n_components=2) #객체 생성
pls2.fit(X, Y) #PLS로 loading, score, weight and coefficient matrix
print("X block loading matrix: ", pls2.x_loadings_)
print("X block score matrix: ", pls2.x_scores_)
print("X block weights matrix: ", pls2.x_weights_)
print("---------------------------------------------------------------------------------------------------------------")
print("Y block loading matrix: ", pls2.y_loadings_)
print("Y block score matrix: ", pls2.y_scores_)
print("Y block weights matrix: ", pls2.y_weights_)
print("---------------------------------------------------------------------------------------------------------------")
print("The coefficient matrix (Y_pls = X*coeff): ", pls2.coef_)

Y_pred = pls2.predict(X)
print("PLS를 통해 차원 줄였다가 복원한 데이터로 재예측한 Y: \n", Y_pred)
# print("Predicted Y value by PLS: ", Y_pred)
# X_score, Y_score = pls2.fit_transform(X, Y)
# print("X score by PLS: ", X_score)
# print("Y score by PLS: ", Y_score)