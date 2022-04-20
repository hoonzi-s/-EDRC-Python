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

#Principal component 1 개 일 때
pca_result1 = PCA(n_components=1) #객체 생성
X_reduced1   = pca_result1.fit_transform(X) #차원 축소 실행
X_recovered1 = pca_result1.inverse_transform(X_reduced1) #차원 축소한 것을 다시 recover

print("원본 데이터: \n", X)
print("PCA를 통해 차원 줄였다가 복원한 데이터: \n", X_recovered1)
print("PCA 결과 각 요소가 차지하는 variance 비율: \n",
      pca_result1.explained_variance_ratio_)

#Principal component 2 개 일 때
pca_result2 = PCA(n_components=2) #객체 생성
X_reduced2   = pca_result2.fit_transform(X) #차원 축소 실행
X_recovered2 = pca_result2.inverse_transform(X_reduced2) #차원 축소한 것을 다시 recover

print("원본 데이터: \n", X)
print("PCA를 통해 차원 줄였다가 복원한 데이터: \n", X_recovered2)
print("PCA 결과 각 요소가 차지하는 variance 비율: \n",
      pca_result2.explained_variance_ratio_)

# X_pca = pca_result2.transform(X)
# print(X_pca)
# print(pca_result2.components_)

#원본 데이터 그림
fig1 = plt.figure(1)
ax  = fig1.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],marker='o')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.title('Raw data in 3d')
plt.show()

#Principal component 영역에서의 데이터 분포
plt.plot(2)
plt.scatter(X_reduced2[:,0],X_reduced2[:,1])
plt.xlabel('pc 1')
plt.ylabel('pc 2')
plt.title('From 3d to 2d (PCA)')
plt.show()

#동일한 그래프 생성 (Principal component 영역에서의 데이터 분포)
# plt.plot(3)
# plt.scatter(X_pca[:,0],X_pca[:,1])
# plt.xlabel('pc 1')
# plt.ylabel('pc 2')
# plt.title('From 3d to 2d (PCA)')
# plt.show()