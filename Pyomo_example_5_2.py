#Planar coordinates (PDE solve)

#convection 추가

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Concrete model 객체 생성
model = ConcreteModel()

# 시뮬레이션 범위
model.x = ContinuousSet(bounds=(0,1))
model.t = ContinuousSet(bounds=(0,3))

# 변수
model.v      = Var(model.t) #v
model.C      = Var(model.t, model.x) #C
model.dCdt   = DerivativeVar(model.C, wrt=model.t) #dT/dt
model.dCdx   = DerivativeVar(model.C, wrt=model.x) #dT/dx
model.d2Cdx2 = DerivativeVar(model.C, wrt=(model.x, model.x)) #d2T/dx2

# Constraint
model.pde_model           = Constraint(model.t, model.x, rule=lambda model, t, x: model.dCdt[t,x] == 0.5*model.d2Cdx2[t,x] - model.v[t]*model.dCdx[t,x] if x > 0 and x < 1 and t > 0 else Constraint.Skip)
model.con1                = Constraint(model.t, rule=lambda model, t:    model.v[t] == 0.1)
model.initial_condition1  = Constraint(model.x, rule=lambda model, x:    model.C[0,x] == 0.5 if x > 0 and x < 1 else Constraint.Skip) #initial condition (at t=0)
model.boundary_condition1 = Constraint(model.t, rule=lambda model, t:    model.C[t,1] == 1)    #boundary condition (at x=1)
model.boundary_condition2 = Constraint(model.t, rule=lambda model, t:    model.dCdx[t,0] == 0) #boundary condition (at x=0)

# @m.Constraint(m.t, m.r)
# def pde(m, t, r):
#     if t == 0:
#         return Constraint.Skip
#     if r == 0 or r == 1:
#         return Constraint.Skip
#     return m.dTdt[t,r] == m.d2Tdr2[t,r]

# Objective function
model.obj = Objective(expr=1) #For just simulation, set as 1

# 적분 구간을 자름
TransformationFactory('dae.finite_difference').apply_to(model, nfe=50, scheme='FORWARD', wrt=model.x)
TransformationFactory('dae.finite_difference').apply_to(model, nfe=50, scheme='FORWARD', wrt=model.t)


#Solver 지정
opt = SolverFactory('ipopt')

#최적화 풀기
print(opt.solve(model, tee=True))
print('---------------------------------------------------------------------------------------------------------------')

# SolverFactory('ipopt').solve(model, tee=True).write()

# 시각화
x = sorted(model.x)
t = sorted(model.t)
v = sorted(model.v)
print(v)
t_mesh = np.zeros((len(t), len(x)))
x_mesh = np.zeros((len(t), len(x)))
C_mesh = np.zeros((len(t), len(x)))

for i in range(0, len(t)):
    for j in range(0, len(x)):
        t_mesh[i,j] = t[i]
        x_mesh[i,j] = x[j]
        C_mesh[i,j] = model.C[t[i],x[j]].value

fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(t_mesh, x_mesh, C_mesh)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('C')
plt.title('PDE solve using pyomo')
plt.show()