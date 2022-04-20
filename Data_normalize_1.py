from sklearn.preprocessing import StandardScaler
import numpy as np

#예제 데이터 생성
np.random.seed(5)
num_data = 50
X=np.empty((num_data, 3))
X[:,0]=3*np.random.rand(num_data) + 2
X[:,1]=5*np.random.rand(num_data) - 1 + 0.2*X[:,0]
X[:,2]=0.3*X[:,0]+2*X[:,1]

#Mean
X_mean1 = np.mean(X)
X_mean2 = np.mean(X,axis=0)
X_mean3 = np.mean(X,axis=1)
print("Matrix 전체 평균: \n", X_mean1)
print("Matrix column 평균: \n", X_mean2)
print("Matrix row 평균: \n", X_mean3)

#Standard deviation
X_std1 = np.std(X)
X_std2 = np.std(X,axis=0)
X_std3 = np.std(X,axis=1)
print("Matrix 전체 standard deviation: \n", X_std1)
print("Matrix column standard deviation: \n", X_std2)
print("Matrix row standard deviation: \n", X_std3)

#Normalize
X_ = (X-X_mean2)/X_std2
print("Normalized Data: \n", X_)
print("X_의 column 평균: \n", np.mean(X_,axis=0))
print("X_의 column standard deviation: \n", np.std(X_,axis=0))

#데이터 복원
X_re = X_*X_std2 + X_mean2
print("복원된 데이터: \n", X_re)