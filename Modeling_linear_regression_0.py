#선형 회귀 분석

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Data
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
x_data = data[:,0].reshape(-1,1) #row data 형태로 만들어 줌.
y_data = data[:,1].reshape(-1,1) #row data 형태로 만들어 줌.

# print(x_data)
x_data_mod = np.insert(x_data, 0, 1, axis=1) #상수 값 부분을 x 축 벡터에 합쳐서 표현
# print(x_data_mod)
# print(x_data_mod.T)

#Analytical solution
coeff = np.linalg.inv(
	x_data_mod.T.dot(x_data_mod)).dot(x_data_mod.T).dot(y_data)
print(coeff)
coeff_a = coeff[1,0]
coeff_b = coeff[0,0]
print("Analytical solution (slope): \n", coeff_a)
print("Analytical solution (bias): \n", coeff_b)
y_predict_analytical = coeff_a*x_data + coeff_b

#Error cost
def Linear_regression_cost(x_data_mod, y_data, p):
	y_predict = np.matmul(x_data_mod,p)
	N = np.size(x_data_mod,0)
	cost = sum((y_data-y_predict)**2)/N
	return cost

cost = Linear_regression_cost(x_data_mod, y_data, [[1],[0.1]])
print("Random selection cost: ", cost)
cost_analytical = Linear_regression_cost(x_data_mod, y_data, [[coeff_b],[coeff_a]])
print("Analtical solution cost: ", cost_analytical)

#Meshgrid 활용해서 Linear regression cost 그래프 그리기
p1 = np.arange(-5, 5, 0.25)
p2 = np.arange(-5, 5, 0.25)
p1_mesh, p2_mesh = np.meshgrid(p1, p2) #meshgrid?
# cost_mesh = Linear_regression_cost(x_data_mod, y_data, [[p1_mesh],[p2_mesh]])
cost_mesh=np.zeros((len(p1),len(p2)))
I=range(len(p1))
J=range(len(p2))
for i in I:
    for j in J:
        p1=p1_mesh[i,j]
        p2=p2_mesh[i,j]
        cost_mesh[i,j] = Linear_regression_cost(x_data_mod, y_data, [[p1],[p2]])
print(cost_mesh)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(p1_mesh, p2_mesh, cost_mesh)
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('Error cost')
plt.title('3D graph example of Error cost')
plt.show()
