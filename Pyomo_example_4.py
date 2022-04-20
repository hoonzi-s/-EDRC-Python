#Dynamic simulation using pyomo (ODE)

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

#상수
k  = 1

#Pyomo model (ODE) 정의
def cstr(X0=0.5):
    # Concrete model 객체 생성
    model = ConcreteModel()

    # 시뮬레이션 범위
    model.t = ContinuousSet(bounds=(0.0, 5.0)) #시뮬레이션 시간의 범위

    # 변수
    model.X = Var(model.t) #변수 X(시간의 함수)
    model.dX = DerivativeVar(model.X) #X의 미분값

    #초기값 (t=0)
    model.X[0.0] = X0

    #Constraint
    model.ode = Constraint(model.t, rule=lambda model, t: model.dX[t] ==  -k* model.X[t])

    return model

#시뮬레이션
tsim, profiles = Simulator(cstr(), package='scipy').simulate(numpoints=100)

#시각화
plt.figure(1)
plt.plot(tsim[0],0.5,'ro',label='Initial')
plt.plot(tsim,profiles,'b--',label='simulation profile')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('X')
plt.title('Simulation using pyomo')
plt.show()