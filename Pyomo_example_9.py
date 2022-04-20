#ODE example (dynamics)

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt

V = 40     # liters
kA = 0.5   # 1/min
kB = 0.1   # l/min
CAf = 2.0  # moles/liter

def batch(X, t):
    CA, CB = X
    dCA_dt = -kA*CA
    dCB_dt = kA*CA - kB*CB
    return [dCA_dt, dCB_dt]

def CB(tf):
    soln = odeint(batch, [CAf, 0], [0, tf])
    return soln[-1][1]

#Optimization
tmax = minimize_scalar(lambda t: -CB(t), bracket=[0,50]).x

print('Concentration c_B has maximum', CB(tmax), 'moles/liter at time', tmax, 'minutes.')

t = np.linspace(0,30,200)
soln = odeint(batch, [CAf,0], t)
plt.plot(t, soln)
plt.xlabel('time / minutes')
plt.ylabel('concentration / moles per liter')
plt.title('Batch Reactor')
plt.legend(['$C_A$','$C_B$'])
plt.grid(True)
plt.show()