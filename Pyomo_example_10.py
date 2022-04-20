#Dynamic optimization using scipy and pyomo (ODE system)

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt

V   = 40    # liters
kA  = 0.5   # 1/min
kB  = 0.1   # l/min
CAf = 2.0   # moles/liter

#---------------------------------------Analytical model--------------------------------------------#
def batch(X, t):
    CA, CB = X
    dCA_dt = -kA*CA
    dCB_dt = kA*CA - kB*CB
    return [dCA_dt, dCB_dt]

#Graph
t = np.linspace(0,30,200)
soln = odeint(batch, [CAf,0], t)
plt.plot(t, soln)
plt.xlabel('time / minutes')
plt.ylabel('concentration / moles per liter')
plt.title('Batch Reactor')
plt.legend(['$C_A$','$C_B$'])
plt.grid(True)
plt.show()
#---------------------------------------------------------------------------------------------------#
#--------------------------------------Analytical optimization--------------------------------------#
def CB(tf):
    soln = odeint(batch, [CAf, 0], [0, tf])
    return soln[-1][1]

tmax = minimize_scalar(lambda t: -CB(t), bracket=[0,50]).x

print('(scipy solution) Concentration c_B has maximum', CB(tmax), 'moles/liter at time', tmax, 'minutes.')
#---------------------------------------------------------------------------------------------------#
#-----------------------------------------Pyomo model-----------------------------------------------#
m = ConcreteModel()

m.tau = ContinuousSet(bounds=(0, 1))

m.tf = Var(domain=NonNegativeReals)
m.cA = Var(m.tau, domain=NonNegativeReals)
m.cB = Var(m.tau, domain=NonNegativeReals)

m.dcA = DerivativeVar(m.cA)
m.dcB = DerivativeVar(m.cB)

m.odeA = Constraint(m.tau, rule=lambda m, tau: m.dcA[tau] == m.tf*(-kA*m.cA[tau]) if tau > 0 else Constraint.Skip)
m.odeB = Constraint(m.tau, rule=lambda m, tau: m.dcB[tau] == m.tf*(kA*m.cA[tau] - kB*m.cB[tau]) if tau > 0 else Constraint.Skip)

m.ic = ConstraintList()   #initial condition
m.ic.add(m.cA[0]  == CAf) #initial condition
m.ic.add(m.cB[0]  == 0)   #initial condition

m.obj = Objective(expr=m.cB[1], sense=maximize)
#---------------------------------------------------------------------------------------------------#
#---------------------------------------Pyomo optimization------------------------------------------#
TransformationFactory('dae.collocation').apply_to(m)
SolverFactory('ipopt').solve(m)
print('(Pyomo solution) Concentration c_B has maximum', m.cB[1](), 'moles/liter at time', m.tf(), 'minutes.')
#---------------------------------------------------------------------------------------------------#