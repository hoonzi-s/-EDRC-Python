#Pyomo modelin & optimization example 1 (Linear)

#Concrete model

from pyomo.environ import *
from pyomo.opt import SolverFactory

#Concrete model 객체 생성
model = ConcreteModel()

#변수
model.x = Var([0,1], domain=NonNegativeReals) #[0,1]은 변수들의 index

#Objective function
model.obj = Objective(expr = 2*model.x[0] + 3*model.x[1])

#Constraint
model.Constraint1 = Constraint(expr = 3*model.x[0] + 4*model.x[1] >= 1)

#Solver 지정
opt = SolverFactory('glpk')

#최적화 풀기
print(opt.solve(model))

#최적화 결과
print("Objective function의 값은: \n", model.obj())
model.display()