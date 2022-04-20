#비모델 기반 최적화 기초 (Numerical derivative-based), Convex

import numpy as np
from scipy.optimize import minimize
from Optimization_objective_1 import obj_convex
import matplotlib.pyplot as plt

#Derivative of the objective function (Numerically)
def derivative_obj1(x):
    h = 10**(-5)
    der = np.zeros_like(x)
    der[0] = (obj_convex([x[0]+h, x[1]])-obj_convex([x[0]-h, x[1]]))/2/h
    der[1] = (obj_convex([x[0], x[1]+h])-obj_convex([x[0], x[1]-h]))/2/h
    return der

#초기값 (Local optimizing algorithm이므로)
initial = np.array([0, 0])

#Gradient descent method로 최적화
x=np.array([[2,1]])
obj_result=np.array(obj_convex(x[0,:]))
step_size=0.001

for i in range(2001):
    new_input = x[i,:]
    new_input = new_input - step_size*derivative_obj1(new_input)
    x=np.vstack((x, new_input))
    obj_result=np.vstack((obj_result,obj_convex(new_input)))

print("Gradient descent method에 따라 계산되는 input 값: ", x)

plt.figure(1)
plt.plot(obj_result)
plt.show()


