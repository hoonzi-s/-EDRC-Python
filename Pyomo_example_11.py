#Parameter estimation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import scipy.optimize as optimize
from pyomo.environ import *

#----------------------------------------------------Data-------------------------------------------------------------#
data = {
    'Expt A': {
        'Expt': 'A',
        'C0'  : 1.64E-4,
        'T'   : [423, 449, 471, 495, 518, 534, 549, 563],
        'C'   : [1.66e-4, 1.66e-4, 1.59e-4, 1.37e-4, 8.90e-5, 5.63e-5, 3.04e-5, 1.71e-5],
    },
    'Expt B': {
        'Expt': 'B',
        'C0'  : 3.69e-4 ,
        'T'   : [423, 446, 469, 490, 507, 523, 539, 553, 575],
        'C'   : [3.73e-4, 3.72e-4, 3.59e-4, 3.26e-4, 2.79e-4, 2.06e-4, 1.27e-4, 7.56e-5, 3.76e-5],
    },
    'Expt C': {
        'Expt': 'C',
        'C0'  : 2.87e-4,
        'T'   : [443, 454, 463, 475, 485, 497, 509, 520, 534, 545, 555, 568],
        'C'   : [2.85e-4, 2.84e-4, 2.84e-4, 2.74e-4, 2.57e-4, 2.38e-4, 2.04e-4, 1.60e-4, 1.12e-4,
                 6.37e-5, 5.07e-5, 4.49e-5],
    },
}

df = pd.concat([pd.DataFrame(data[expt]) for expt in data])
df = df.set_index('Expt')
print(df)
#---------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------Graph-----------------------------------------------------------#
plt.figure(1)
for expt in sorted(set(df.index)):
    plt.scatter(df['T'][expt], df['C'][expt])
plt.ylim(0, 1.1 * max(df['C']))
plt.xlabel('temperature / K')
plt.ylabel('concentration')
plt.legend(["Expt. " + expt for expt in sorted(set(df.index))])
plt.title('Effluent Concentrations');
plt.grid(True)
plt.show()

# add a column 'X' to the dataframe
df['X'] = 1 - df['C']/df['C0']
plt.figure(2)
for expt in sorted(set(df.index)):
    plt.scatter(df['T'][expt], df['X'][expt])
plt.xlabel('Temperature (K)')
plt.ylabel('conversion ratios')
plt.legend(['C0 = ' + str(list(df['C0'][expt])[0]) for expt in sorted(set(df.index))])
plt.title('Conversion at different feed concentrations')
plt.grid(True)
plt.show()
#---------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------Model------------------------------------------------------------#
Tr = 298    # reference temperature.
q = 0.1     # flow rate (liters/min)
m = 1       # amount of catalyst (g)

#Error by parameter guess
def residuals(parameters, df):
    n, lnk0, ERTr = parameters
    C0, C, T = df['C0'], df['C'], df['T']
    return C0 - C - (m/q) * C**n  * (T/Tr)**n * np.exp(lnk0 - ERTr*Tr/T)

#Initial guesses
parameter_names = ['n', 'lnk0', 'ERTr']
parameter_guess = [1, 15, 38]

#Error from the initial guesses
def plot_residuals(r, df, ax=None):
    rmax = np.max(np.abs(r))
    if ax is None:
        fig, ax = plt.subplots(1, len(df.columns), figsize=(12,3))
    else:
        rmax = max(ax[0].get_ylim()[1], rmax)
    n = 0
    for c in df.columns:
        ax[n].scatter(df[c], r)
        ax[n].set_ylim(-rmax, rmax)
        ax[n].set_xlim(min(df[c]), max(df[c]))
        ax[n].plot(ax[n].get_xlim(), [0,0], 'r')
        ax[n].set_xlabel(c)
        ax[n].set_title('Residuals')
        ax[n].grid(True)
        n += 1
    plt.tight_layout()

#Execution!
r = residuals(parameter_guess, df) #Parameter guess
plot_residuals(r, df)
plt.show()
#---------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------Parameter estimation--------------------------------------------------#
def sos(parameters, df): #Objective function in the parameter estimation
    return sum(r**2 for r in residuals(parameters, df))

def best_fit(fcn, df, disp=1): #Minimize the above objective function
    return optimize.fmin(fcn, parameter_guess, args=(df,), disp=disp)

parameter_fit = best_fit(sos, df) #optimized parameters

for name,value in zip(parameter_names, parameter_fit):
    print(name, " = ", round(value,2))

r = residuals(parameter_fit, df) #Optimized parameters
plot_residuals(r, df);
plt.show()

n, lnk0, ERTr = parameter_fit
for expt in set(df.index):
    y = (df['C0'][expt] - df['C'][expt])/df['C'][expt]**n
    x = (m/q) * (df['T'][expt]/Tr)**n * np.exp(lnk0 - ERTr*Tr/df['T'][expt])
    plt.plot(x, y, marker='o', lw=0)
plt.plot(plt.xlim(), plt.ylim())
plt.ylabel('$y_k$')
plt.xlabel('$x_k$')
plt.legend(set(df.index))
plt.grid(True)
plt.show()

for expt in set(df.index):
    y = df['C0'][expt]**(n-1) * (m/q) * (df['T'][expt]/Tr)**n * np.exp(lnk0 - ERTr*Tr/df['T'][expt])
    x = df['X'][expt]/(1 - df['X'][expt])**n
    plt.plot(x, y, marker='o', lw=0)
plt.plot(plt.xlim(), plt.ylim())
plt.xlabel('')
plt.grid(True)
plt.show()
#---------------------------------------------------------------------------------------------------------------------#