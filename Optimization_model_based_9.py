#모델 기반 최적화 (Nonconvex, constrained) using 'Sequential least squares programming Algirhtm'
#objective function에 parameter 값 넣기

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import SR1
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Nonconvex objective function model
def obj2(x,p):
    return -(p*x[1] + 20) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 20))))-x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 20))))

#Convex nonlinear constraint (c(x)>=0)
def Nonlinear_con1(x):
    return [x[0] +x[1] - 100, x[1]**2 - 20]

ineq_cons = {'type': 'ineq',
           'fun': Nonlinear_con1}

#초기값 (Local optimizing algorithm이므로)
initial = np.array([100, 100], dtype='float64')

#parameter 값
p=2
result_1 = minimize(obj2, initial, method='SLSQP', bounds=None, constraints=ineq_cons, args=(p))

print("Optimal input: \n", result_1.x), print("-----------------------------------------------------------------------")
print("Optimal output: \n", result_1.fun), print("-----------------------------------------------------------------------")

#그래프 그리기
x1 = np.arange(-500, 500)
x2 = np.arange(-500, 500)
x1_grid, x2_grid = np.meshgrid(x1,x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, obj2([x1_grid, x2_grid],p), cmap='Wistia')
ax.scatter(result_1.x[0], result_1.x[1],result_1.fun, marker='o', c='r')
plt.show()