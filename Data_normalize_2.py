from sklearn.preprocessing import StandardScaler
import numpy as np

#예제 데이터 생성
np.random.seed(5)
num_data = 50
X=np.empty((num_data, 3))
X[:,0]=3*np.random.rand(num_data) + 2
X[:,1]=5*np.random.rand(num_data) - 1 + 0.2*X[:,0]
X[:,2]=0.3*X[:,0]+2*X[:,1]

#Normalize
X_ = StandardScaler().fit_transform(X) #X_=(X-mean(X))/std(X)

#확인
print("X_의 column 평균: \n", np.mean(X_,axis=0))
print("X_의 column standard deviation: \n", np.std(X_,axis=0))