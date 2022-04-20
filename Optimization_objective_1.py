#Plant equation for the no_model_based optimization
#Unknown
#Convex, Nonconvex

#Objective function 1 (Convex)
def obj_convex(x):
    return (x[0]-100)**2 + (x[1]+20)**2

#Objective function 2 (Nonconvex)
def obj_nonconvex(x):
    return x