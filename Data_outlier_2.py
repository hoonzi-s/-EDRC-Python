import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def distance_sort(x_new,x_data):
    error=np.array(x_new-x_data)
    error_norm=np.linalg.norm(error,axis=1)
    x_data_new_pandas=pd.DataFrame({'error':error_norm,
                                    'x1':x_data[:,0],
                                    'x2':x_data[:,1]})
    x_data_sort=x_data_new_pandas.sort_values(by='error')
    return x_data_sort

def KNN_example(x_new,x_data,K):
    x_data_sort=distance_sort(x_new,x_data)
    K_nearest_x_data_pandas=x_data_sort.head(K)
    K_nearest_x_data=K_nearest_x_data_pandas[["x1","x2"]].values
    return K_nearest_x_data

x_new=np.array([1.5,2])
x_data=np.array([[1,0.5],
                 [3,3],
                 [2,2],
                 [3,2],
                 [0,1]])

print(distance_sort(x_new,x_data))

print("주어진 데이터 중 x_new에 가장 가까운 2개: \n", KNN_example(x_new,x_data,2))

plt.figure(1)
plt.scatter(x_data[:,0],x_data[:,1],label='Existing data')
plt.scatter(x_new[0],x_new[1],label='New data')
plt.legend(loc='upper left')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('K nearest neighborhood example')
plt.show()