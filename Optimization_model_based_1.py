#모델 기반 최적화 기초 (Convex, Unconstrained)
#Gradient 정보 없이

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

#Convex objective function model
def obj1(x):
    return (x[0]-100)**2 + (x[1]+20)**2

#초기값 (Local optimizing algorithm이므로)
initial = np.array([0, 0])

#최적화 1 using Nelder-Mead simplex method
result_1 = minimize(obj1, initial, method='nelder-mead') #simplex

#결과
print("Optimal input: \n", result_1.x), print("--------------------------------------------------------------------------")
print("Optimal output:\n", result_1.fun), print("--------------------------------------------------------------------------")

#최적화 2 using differential evolution method
bounds2 = [(-200,200), (-200, 200)] #boundary
result_2 = differential_evolution(obj1, bounds2)
print("Optimal input: \n", result_2.x), print("--------------------------------------------------------------------------")
print("Optimal output:\n", result_2.fun), print("--------------------------------------------------------------------------")