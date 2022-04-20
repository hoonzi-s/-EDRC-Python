#비모델 기반 최적화 기초 (Derivative-free), Convex
#Differential evolution (Genetic algorithm과 같은 Evolutionary computation의 일종) 알고리즘 이용

import numpy as np
from scipy.optimize import differential_evolution
from Optimization_objective_1 import obj_convex

#최적화 1 (bound 좁을 때 )
bounds1=[(-10,10),(-10,10)] #boundary
result_1 = differential_evolution(obj_convex, bounds1)

#결과
print("종합 정보: \n", result_1), print("--------------------------------------------------------------------------")
print("Optimal input: \n", result_1.x), print("--------------------------------------------------------------------------")
print("Optimal output:\n", result_1.fun), print("--------------------------------------------------------------------------")

#최적화 2 (bound 넓을 때)
bounds2 = [(-200,200), (-200, 200)] #boundary
result_2 = differential_evolution(obj_convex, bounds2)
print("Optimal input: \n", result_2.x), print("--------------------------------------------------------------------------")
print("Optimal output:\n", result_2.fun), print("--------------------------------------------------------------------------")

#최적화 3 (options 교체)
#maxiter: The maximum number of generations over which the entire population is evolved.
#popsize: A multiplier for setting the total population size.
#tol: Tolerance (Threshold value) for the convergence
#mutation:
#recombination: Increasing this value allows a larger number of mutants to progress into the next generation (population stability down)
#seed: Initial points