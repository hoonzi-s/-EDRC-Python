#비모델 기반 최적화 기초 (Derivative-free), Convex

import numpy as np
from scipy.optimize import minimize
from Optimization_objective_1 import obj_convex
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#objective function 그림
x1 = np.arange(-200,201)
x2 = np.arange(-200,201)
x1_grid, x2_grid = np.meshgrid(x1, x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, obj_convex([x1_grid,x2_grid]))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('obj_convex')
plt.show()


#초기값 (Local optimizing algorithm이므로)
initial = np.array([0, 0])

#최적화
result1 = minimize(obj_convex, initial, method='nelder-mead') #simplex

#결과
print("종합 정보: \n", result1)
print("Optimal input: \n", result1.x)
print("Optimal output:\n", result1.fun)