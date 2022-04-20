#모델 기반 최적화 기초 (Nonconvex, Unconstrained)
#Gradient 정보 이용

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Nonconvex objective function model
def obj2(x):
    return -(x[1] + 20) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 20))))-x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 20))))

#초기값 (Local optimizing algorithm이므로)
initial = np.array([0, 0])

#Derivative 정보 안 줬을 때: 알아서 numerical하게 계산해 줌. 그러므로 사실은 derivative_obj_convex(x)도 정의할 필요 없었다.
result_2 = minimize(obj2, initial, method='BFGS', options={'disp':True})
print("종합 정보: \n", result_2)
print("Optimal input: \n", result_2.x)
print("Optimal output: \n", result_2.fun)