#Dynamic simulation using pyomo (ODE)

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

#상수
k11 = 1
k12 = 0.3
k21 = 0.5
k22 = 1.1
X1_initial=0.5
X2_initial=0.3

#Pyomo model (ODE) 정의
def cstr(X10=X1_initial, X20=X2_initial):
    # Concrete model 객체 생성
    model = ConcreteModel()

    # 시뮬레이션 범위
    model.t = ContinuousSet(bounds=(0.0, 10.0)) #시뮬레이션 시간의 범위

    # 변수
    model.X1 = Var(model.t)  #변수 X1 (시간의 함수)
    model.X2 = Var(model.t)  #변수 X1 (시간의 함수)
    model.dX1 = DerivativeVar(model.X1) #X1의 미분값
    model.dX2 = DerivativeVar(model.X2) #X2의 미분값

    #초기값 (t=0)
    model.X1[0.0] = X10
    model.X2[0.0] = X20

    #Constraint
    model.ode1 = Constraint(model.t, rule=lambda model, t: model.dX1[t] == -k11*model.X1[t])
    model.ode2 = Constraint(model.t, rule=lambda model, t: model.dX2[t] == -k21*model.X2[t]+k22*model.X1[t])

    return model

#시뮬레이션
tsim, profiles = Simulator(cstr(), package='scipy').simulate(numpoints=100)

#시각화
plt.figure(1)
plt.plot(0,X1_initial,'bo',label='Initial of X1')
plt.plot(0,X2_initial,'ro',label='Initial of X2')
plt.plot(tsim, profiles[:,0],'b-',label='Simulation profile of X1')
plt.plot(tsim, profiles[:,1],'r-',label='Simulation profile of X2')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('X')
plt.title('Simulation using pyomo')
plt.show()