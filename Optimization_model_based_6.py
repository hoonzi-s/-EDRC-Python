#모델 기반 최적화 (Convex, constrained) using 'Sequential least squares programming Algirhtm'

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

#Convex nonlinear constraint (c(x)>=0)
def Nonlinear_con1(x):
    return [x[0] +x[1] - 100, x[1]**2 - 20]

ineq_cons = {'type': 'ineq',
           'fun': Nonlinear_con1}

#초기값 (Local optimizing algorithm이므로)
initial = np.array([100, 100], dtype='float64')

result_1 = minimize(obj1, initial, method='SLSQP', bounds=None, constraints=ineq_cons)

print("Optimal input: \n", result_1.x), print("-----------------------------------------------------------------------")
print("Optimal output: \n", result_1.fun), print("-----------------------------------------------------------------------")
