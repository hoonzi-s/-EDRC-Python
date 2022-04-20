#모델 기반 최적화 (Nonconvex, constrained) using 'Trust-Region Constrained Algirhtm'

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
def obj2(x):
    return -(x[1] + 20) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 20))))-x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 20))))

#Bounds
bounds = Bounds([-np.inf, -10000], [np.inf, 10000])

#Convex linear constraint
linear_con = LinearConstraint([[1,0], [0,1]], [-100000, -10000], [10000, 10000])

#Convex nonlinear constraint (lb<=c(x)<=ub)
def Nonlinear_con1(x):
    return [-x[0] - x[1] + 100., -x[1]**2 + 20.]

nonlinear_con = NonlinearConstraint(Nonlinear_con1, -np.inf, 0, jac='2-point', hess=BFGS())

#최적화 1 using Trust-Region Constrained Algorithm + starting point: initial_1
initial_1 = np.array([100, 100], dtype='float64') #초기값 (Local optimizing algorithm이므로)
result_1 = minimize(obj2, initial_1, method='trust-constr', jac="2-point", hess=SR1(),
                    constraints=[linear_con, nonlinear_con],
                    options={'verbose': 1}, bounds=None)
print("Optimal input (starting point is intial_1): \n", result_1.x), print("---------------------------------------")
print("Optimal output (starting point is intial_1): \n", result_1.fun), print("--------------------------------------")

#최적화 2 using Trust-Region Constrained Algorithm + starting point: initial_2
initial_2 = np.array([50, 200], dtype='float64') #초기값 (Local optimizing algorithm이므로)
result_2 = minimize(obj2, initial_2, method='trust-constr', jac="2-point", hess=SR1(),
                    constraints=[linear_con, nonlinear_con],
                    options={'verbose': 1}, bounds=None)
print("Optimal input (starting point is intial_2): \n", result_2.x), print("---------------------------------------")
print("Optimal output (starting point is intial_2): \n", result_2.fun), print("--------------------------------------")

#그래프 그리기
x1 = np.arange(-200, 201)
x2 = np.arange(-200, 201)
x1_grid, x2_grid = np.meshgrid(x1,x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, obj2([x1_grid, x2_grid]), cmap='Wistia')
ax.scatter(result_1.x[0], result_1.x[1],result_1.fun, marker='o', c='r')
ax.scatter(result_2.x[0], result_2.x[1],result_2.fun, marker='o', c='k')
plt.show()