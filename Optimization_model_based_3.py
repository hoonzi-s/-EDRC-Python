#모델 기반 최적화 기초 (Nonconvex, Unconstrained)
#Gradient 정보 없이

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Nonconvex objective function model
def obj2(x):
    return -(x[1] + 20) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 20))))-x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 20))))

#그래프 그리기
x1 = np.arange(-200, 201)
x2 = np.arange(-200, 201)
x1_grid, x2_grid = np.meshgrid(x1,x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, obj2([x1_grid, x2_grid]))
plt.show()

#초기값 (Local optimizing algorithm이므로)
initial = np.array([20, 20])

#최적화 1 using Nelder-Mead simplex method
result_1 = minimize(obj2, initial, method='nelder-mead') #simplex

#결과
print("Optimal input by Nelder_Mead simplex method: \n", result_1.x), print("--------------------------------------------------------------------------")
print("Optimal output by Nelder_Mead simplex method:\n", result_1.fun), print("--------------------------------------------------------------------------")

#최적화 2 using differential evolution method
bounds2 = [(-50,50), (-50, 50)] #boundary
result_2 = differential_evolution(obj2, bounds2)
print("Optimal input by differential evolution method: \n", result_2.x), print("--------------------------------------------------------------------------")
print("Optimal output by differential evolution method: \n", result_2.fun), print("--------------------------------------------------------------------------")

#그래프 그리기
x1 = np.arange(-50, 51)
x2 = np.arange(-50, 51)
x1_grid, x2_grid = np.meshgrid(x1,x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, obj2([x1_grid, x2_grid]), cmap='binary')
ax.scatter(result_1.x[0], result_1.x[1],result_1.fun, marker='o', c='r')
ax.scatter(result_2.x[0], result_2.x[1],result_2.fun, marker='o', c='b')
plt.show()