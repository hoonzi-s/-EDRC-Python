#User-defined package 예제

from Optimization_objective_1 import obj_convex
from Optimization_objective_1 import obj_nonconvex

result1=obj_convex([1,1])
print("User-defined package에서 가져온 함수 obj_convex의 결과 값은: ", result1)

result2=obj_nonconvex([[1],[2]])
print("User-defined package에서 가져온 함수 obj_nonconvex의 결과 값은: ", result2)