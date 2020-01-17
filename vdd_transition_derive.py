from sympy import *
from sympy.stats import Normal, density, E, variance

var("tau nact pact dt std damping scale tau_threshold act_threshold", real=True)
var("dt std alpha scale time_constant", real=True, positive=True)
noise = Normal("noise", 0, std)

alpha = 1 - exp(-dt/time_constant)
#nact = alpha*pact + (1 - alpha)*(atan((tau - tau_threshold)*scale) + noise)
#dact = (nact - pact).simplify()
#pprint(dact)
#nact = pact + (-damping*pact + atan((tau - tau_threshold)*scale) + noise)*dt
dact = -alpha*pact + atan((tau - tau_threshold)*scale)*dt + noise*sqrt(dt)
print(dact)
pprint(density(dact))
m = E(dact)
v = variance(dact).simplify()
print(m)
print(v)
