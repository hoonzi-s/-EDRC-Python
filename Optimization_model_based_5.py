#모델 기반 최적화 (Convex, constrained) using 'Trust-Region Constrained Algorithm'

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import BFGS
from scipy.optimize import SR1
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds

#Convex objective function model
def obj1(x):
    return (x[0]-100)**2 + (x[1]+20)**2

#Bounds
bounds = Bounds([-20, -10], [10, 10])

#Convex linear constraint
linear_con = LinearConstraint([[1,0], [0,1]], [-100000, -10000], [10000, 10000])

#Convex nonlinear constraint (lb<=c(x)<=ub)
def Nonlinear_con1(x):
    return [-x[0] - x[1] + 100., -x[1]**2 + 20.]

nonlinear_con = NonlinearConstraint(Nonlinear_con1, -np.inf, 0, jac='2-point', hess=BFGS())

#초기값 (Local optimizing algorithm이므로)
initial = np.array([100, 100], dtype='float64')

#최적화 1 using Trust-Region Constrained Algorithm (Linear constraint만 있을 때)
result_1 = minimize(obj1, initial, method='trust-constr', jac="2-point", hess=SR1(),
                    constraints=[linear_con],
                    options={'verbose': 1}, bounds=None)
print("Optimal input: \n", result_1.x), print("----------------------------------------------------------------")
print("Optimal output: \n", result_1.fun), print("----------------------------------------------------------------")

#최적화 2 using Trust-Region Constrained Algorithm (Nonlinear constraint만 있을 때)
result_2 = minimize(obj1, initial, method='trust-constr', jac="2-point", hess=SR1(),
                    constraints=[nonlinear_con],
                    options={'verbose': 1}, bounds=None)
print("Optimal input: \n", result_2.x), print("----------------------------------------------------------------")
print("Optimal output: \n", result_2.fun), print("----------------------------------------------------------------")

#최적화 3 using Trust-Region Constrained Algorithm (Linear, Nonlinear constraint 둘 다 있을 때)
result_3 = minimize(obj1, initial, method='trust-constr', jac="2-point", hess=SR1(),
                    constraints=[linear_con, nonlinear_con],
                    options={'verbose': 1}, bounds=None)
print("Optimal input: \n", result_3.x), print("----------------------------------------------------------------")
print("Optimal output: \n", result_3.fun), print("----------------------------------------------------------------")

#최적화 4 using Trust-Region Constrained Algorithm (Linear, Nonlinear constraint 둘 다 있을 때 + bound)
result_4 = minimize(obj1, initial, method='trust-constr', jac="2-point", hess=SR1(),
                    constraints=[linear_con, nonlinear_con],
                    options={'verbose': 1}, bounds=bounds)
print("Optimal input: \n", result_4.x), print("----------------------------------------------------------------")
print("Optimal output: \n", result_4.fun), print("----------------------------------------------------------------")