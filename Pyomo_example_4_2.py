#Dynamic simulation using pyomo (ODE)

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

Ea = 72750  # activation energy J/gmol
R = 8.314  # gas constant J/gmol/K
k0 = 7.2e10  # Arrhenius rate constant 1/min
V = 100.0  # Volume [L]
rho = 1000.0  # Density [g/L]
Cp = 0.239  # Heat capacity [J/g/K]
dHr = -5.0e4  # Enthalpy of reaction [J/mol]
UA = 5.0e4  # Heat transfer [J/min/K]
q = 100.0  # Flowrate [L/min]
cAi = 1.0  # Inlet feed concentration [mol/L]
Ti = 350.0  # Inlet feed temperature [K]
cA0 = 0.5  # Initial concentration [mol/L]
T0 = 350.0  # Initial temperature [K]
Tc = 300.0  # Coolant temperature [K]

#Pyomo model for CSTR
def cstr(cA0=0.5, T0=350.0):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0.0, 10.0))
    m.cA = Var(m.t)
    m.T = Var(m.t)
    m.dcA = DerivativeVar(m.cA) #dCA/dt
    m.dT = DerivativeVar(m.T) #dT/dt

    # Setting the initial conditions
    m.cA[0.0] = cA0
    m.T[0.0] = T0

    k = lambda T: k0 * exp(-Ea / R / T)
    m.ode1 = Constraint(m.t, rule=lambda m, t: V * m.dcA[t] == q * (cAi - m.cA[t]) - V * k(m.T[t]) * m.cA[t])
    m.ode2 = Constraint(m.t, rule=lambda m, t: V * rho * Cp * m.dT[t] == q * rho * Cp * (Ti - m.T[t]) + (-dHr) * V * k(m.T[t]) * m.cA[t] + UA * (Tc - m.T[t]))

    return m

#Simulation
tsim, profiles = Simulator(cstr(), package='scipy').simulate(numpoints=100)

#시각화
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(tsim, profiles[:,0],'b-')
plt.xlabel('t')
plt.ylabel('X')
plt.title('Simulation using pyomo (X)')

plt.subplot(1,2,2)
plt.plot(tsim, profiles[:,1],'r-')
plt.xlabel('t')
plt.ylabel('Temperature')
plt.title('Simulation using pyomo (T)')
plt.show()
