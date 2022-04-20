#not ODE but static system optimization (analytically & numerically)

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

V = 40     # liters
kA = 0.5   # 1/min
kB = 0.1   # l/min
CAf = 2.0  # moles/liter

def cstr(q):
    return q*V*kA*CAf/(q + V*kB)/(q + V*kA)

#---------------------------------------Analytical solution----------------------------------------#
qmax = V*np.sqrt(kA*kB)
CBmax = cstr(qmax)
print('(Analytical) Flowrate at maximum CB = ', qmax, 'liters per minute.')
print('(Analytical) Maximum CB =', CBmax, 'moles per liter.')
print('(Analytical) Productivity = ', qmax*CBmax, 'moles per minute.')
#--------------------------------------------------------------------------------------------------#

print("===========================================================================")

#----------------------------------------Numerical solution----------------------------------------#
# create a model instance
m = ConcreteModel()

# create the decision variable
m.q = Var(domain=NonNegativeReals)

# create the objective
m.CBmax = Objective(expr=m.q*V*kA*CAf/(m.q + V*kB)/(m.q + V*kA), sense=maximize)

# solve using the nonlinear solver ipopt
SolverFactory('ipopt').solve(m)

# print solution
print('(Numerical) Flowrate at maximum CB = ', m.q(), 'liters per minute.')
print('(Numerical) Maximum CB =', m.CBmax(), 'moles per liter.')
print('(Numerical) Productivity = ', m.q()*m.CBmax(), 'moles per minute.')
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------Figure------------------------------------------------#
plt.figure(1)
q = np.linspace(0,30,200)
plt.plot(q, cstr(q))
plt.xlabel('flowrate q / liters per minute')
plt.ylabel('concentration C_B / moles per liter')
plt.title('Outlet concentration for a CSTR')
plt.grid(True)
plt.show()
#--------------------------------------------------------------------------------------------------#