#Principal component analysis (PCA) 기초 (Data reduction)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#예제 데이터 생성
np.random.seed(5)
num_data = 50
X=np.empty((num_data, 3))
X[:,0]=3*np.random.rand(num_data) + 2
X[:,1]=5*np.random.rand(num_data) - 1 + 0.2*X[:,0]
X[:,2]=0.3*X[:,0]+2*X[:,1]


#데이터 표준화 (Normalize, 평균 =0, 표준편차 1)
mean_X=np.mean(X,axis=0)
std_X=np.std(X,axis=0)
X_ = StandardScaler().fit_transform(X) #X_=(X-mean(X))/std(X)
# X_re = X_*std_X+mean_X (복원)

#Principal component 1 개 일 때 + normalized data 사용 했을 때
pca_result3 = PCA(n_components=1) #객체 생성
X_reduced3_   = pca_result3.fit_transform(X_) #차원 축소 실행
X_recovered3_ = pca_result3.inverse_transform(X_reduced3_) #차원 축소한 것을 다시 recover
X_recovered3  = X_recovered3_*std_X + mean_X #normalized 했던거 다시 recover

print("원본 데이터: \n", X)
print("PCA를 통해 차원 줄였다가 복원한 데이터: \n", X_recovered3)
print("PCA 결과 각 요소가 차지하는 variance 비율: \n",
      pca_result3.explained_variance_ratio_)

#Principal component 2 개 일 때 + normalized data 사용 했을 때
pca_result4 = PCA(n_components=2) #객체 생성
X_reduced4_   = pca_result4.fit_transform(X_) #차원 축소 실행
X_recovered4_ = pca_result4.inverse_transform(X_reduced4_) #차원 축소한 것을 다시 recover
X_recovered4  = X_recovered4_*std_X + mean_X #normalized 했던거 다시 recover

print("원본 데이터: \n", X)
print("PCA를 통해 차원 줄였다가 복원한 데이터: \n", X_recovered4)
print("PCA 결과 각 요소가 차지하는 variance 비율: \n",
      pca_result4.explained_variance_ratio_)

#Principal component 영역에서의 데이터 분포
plt.plot(2)
plt.scatter(X_reduced4_[:,0],X_reduced4_[:,1])
plt.xlabel('pc 1')
plt.ylabel('pc 2')
plt.title('From 3d to 2d (PCA) + normalize (preprocessing)')
plt.show()