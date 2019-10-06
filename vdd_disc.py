import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit, njit

@njit(parallel=True)
def normpdf(x, m, v):
    return np.exp(-(m - x)**2/(2*v))/np.sqrt(2*np.pi*v)

@njit(parallel=True)
def normcdf(x, m, v):
    return (1 + math.erf((x - m)/np.sqrt(v*2)))/2

@njit(parallel=True)
def normsf(*args):
    return 1.0 - normcdf(*args)

@njit(parallel=True)
def vdd_step(tau, pacts, pweights, nweights, dt, std, damping, scale, tau_threshold, act_threshold):
    dead = 0.0
    N = len(pacts)
    for i in range(N):
        pact = pacts[i]
        alpha = 1 - np.exp(-dt/damping)
        diff_mean = dt*(-alpha*pact + np.arctan(scale*(tau - tau_threshold)))
        diff_var = dt*std**2
        dead += normsf(act_threshold - pact, diff_mean, diff_var)*pweights[i]
        
        for j in range(N):
            diff = pacts[j] - pact
            diff_prob = normpdf(diff, diff_mean, diff_var)
            nweights[j] += diff_prob*pweights[i]
    nweights /= np.sum(nweights)
    return dead

@njit
def vdd_pdf(taus, dt, std, damping, scale, tau_threshold, act_threshold, N=100, minact=-5):
    pacts = np.linspace(minact, act_threshold, N)
    pweights = np.zeros_like(pacts)
    pweights[np.searchsorted(pacts, 0.0)] = 1.0
    pweights /= np.sum(pweights)
    nweights = np.zeros_like(pweights)
    #cumalive = np.empty(len(taus))
    deadpdf = np.empty(len(taus))
    alive = 1.0

    for i, tau in enumerate(taus):
        dead = alive*vdd_step(tau, pacts, pweights, nweights,
                dt, std, damping, scale, tau_threshold, act_threshold
                )
        alive -= dead
        #cumalive[i] = alive
        deadpdf[i] = dead
        
        pweights, nweights = nweights, pweights
        nweights[:] = 0.0
    
    return deadpdf/dt

def test():
    dt = 1/30
    dur = 20
    ts = np.arange(0, dur, dt)
    
    tau0 = 3.5
    speed = 20.0
    dist = tau0*speed - ts*speed
    tau = dist/speed

    hacktau = tau.copy()
    hacktau[hacktau < 0] = 1e50
    
    for std in np.linspace(0.1, 2.0, 10):
        param = dict(
            std=std,
            damping=0.3,
            scale=1.0,
            tau_threshold=3.5,
            act_threshold=0.5,
            )

        pdf = vdd_pdf(hacktau, dt, **param)
        plt.plot(ts, pdf)
    plt.show()

if __name__ == '__main__':
    test()

