import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit, njit, prange, vectorize, float64
from collections import namedtuple
#def njit(*args, **kwargs):
#    return lambda f: f

VddmParams = namedtuple('VddmParams',
    'dt     std     damping     tau_threshold   pass_threshold  scale   act_threshold',
    defaults=(                                  0.0,            1.0,    1.0)
        )

eps = np.finfo(float).eps

DEFAULT_MINACT = -3.0
DEFAULT_MAXACT = 3.0

@njit()
def normpdf(x, m, v):
    return np.exp(-(x - m)**2/(2*v))/np.sqrt(2*np.pi*v)

@vectorize()
def normcdf(x, m, v):
    return (1.0 + math.erf((x - m)/np.sqrt(v*2)))/2.0

@vectorize()
def stdnormcdf(x):
    return 0.5 + math.erf(x)/2.0

@njit()
def normsf(*args):
    return 1.0 - normcdf(*args)

@njit(parallel=False, fastmath=True)
def vdd_step(p, da, tau, acts, pweights, nweights, decision_prob=1.0):
    decided = 0.0
    N = len(acts)
    dt = p.dt; std=p.std; damping=p.damping;
    tau_threshold=p.tau_threshold; scale=p.scale; act_threshold=p.act_threshold
    
    alpha = 1 - np.exp(-dt*damping)
    diff_var = dt*std**2
    if tau >= p.pass_threshold:
        diff_mean_tau = dt*np.arctan(scale*(tau - tau_threshold))
    else:
        diff_mean_tau = dt*np.pi/2
    
    hacks = math.sqrt(2.0*diff_var)
    for fr in range(N):
        too_small = 0.0
        frw = pweights[fr]
        act_fr = acts[fr]
        diff_mean = diff_mean_tau - alpha*act_fr
        for to in range(N - 1):
            diff = acts[to] - act_fr
            small_enough = stdnormcdf((diff + da/2 - diff_mean)/hacks)
            nweights[to] += (small_enough - too_small)*frw
            too_small = small_enough
        nweights[-1] += (1.0 - small_enough)*frw
    
    # TODO: No need to do the whole loop, but one more loop here
    # doesn't matter so much
    for i in range(N):
        a = acts[i]
        # Assume the activations are uniformly distributed within a
        # bin. Important for getting continuous derivatives w.r.t. act_threshold
        share_over = (a - p.act_threshold)/da + 0.5
        share_over = min(max(share_over, 0.0), 1.0)
        bindec = decision_prob*share_over*nweights[i]
        nweights[i] -= bindec
        decided += bindec
    nweights /= np.sum(nweights)
    return decided

@njit()
def vdd_activation_pdf(p, taus, N=100, minact=DEFAULT_MINACT, maxact=DEFAULT_MAXACT):
    acts = np.linspace(minact, maxact, N)
    da = acts[1] - acts[0]
    weights = np.zeros((len(taus) + 1, N))
    weights[0, np.searchsorted(acts, 0.0)] = 1.0
    
    deadpdf = np.empty(len(taus))
    alive = 1.0

    for i, tau in enumerate(taus):
        dead = alive*vdd_step(p, da, tau, acts, weights[i], weights[i+1])
        alive -= dead
        deadpdf[i] = dead
        
    return weights[1:], deadpdf/p.dt, acts

@njit()
def vdd_decision_pdf(p, taus, N=100, minact=DEFAULT_MINACT, maxact=DEFAULT_MAXACT):
    acts = np.linspace(minact, maxact, N)
    da = acts[1] - acts[0]

    pweights = np.zeros_like(acts)
    pweights[np.searchsorted(acts, 0.0)] = 1.0
    nweights = np.zeros_like(pweights)
    
    deadpdf = np.empty(len(taus))
    alive = 1.0

    for i, tau in enumerate(taus):
        dead = alive*vdd_step(p, da, tau, acts, pweights, nweights)
        pweights, nweights = nweights, pweights
        nweights[:] = 0

        alive -= dead
        deadpdf[i] = dead
        
    return deadpdf/p.dt

@njit()
def vdd_blocker_activation_pdf(p, taus, blocker_taus, N=100, minact=DEFAULT_MINACT, maxact=DEFAULT_MAXACT):
    acts = np.linspace(minact, maxact, N)
    da = acts[1] - acts[0]
    weights = np.zeros((len(taus) + 1, N))
    weights[0, np.searchsorted(acts, 0.0)] = 1.0
    bweights = weights.copy()
    
    crossedpdf = np.empty(len(taus))
    unblockedpdf = np.empty(len(taus))
    uncrossed = 1.0
    blocked = 1.0

    for i, (tau, btau) in enumerate(zip(taus, blocker_taus)):
        gone = 1.0 if btau <= p.pass_threshold else 0.0
        unblocked = blocked*vdd_step(p, da, btau, acts, bweights[i], bweights[i+1], decision_prob=gone)
        blocked -= unblocked
        
        crossed = uncrossed*vdd_step(p, da, tau, acts, weights[i], weights[i+1], 1.0 - blocked)
        uncrossed -= crossed
        
        crossedpdf[i] = crossed
        unblockedpdf[i] = unblocked
        
    return (weights[1:], bweights[1:]), (crossedpdf/p.dt, unblockedpdf/p.dt), acts


def vdd_loss(trials, dt, N=100):
    taus, rts = zip(*trials)
    ts = [np.arange(len(tau))*dt for tau in taus]
    rtis = [t.searchsorted(rt) for (t, rt) in zip(ts, rts)]


    def loss(**kwargs):
        lik = 0
        for tau, rti in zip(taus, rtis):
            #pdf = vdd_decision_pdf(tau, dt, **kwargs, N=N)
            kwargs['dt'] = dt
            pdf = vdd_decision_pdf(VddmParams(**kwargs), tau, N=N)
            lik += np.sum(np.log(pdf[rti] + eps))

        return -lik
    
    return loss

def test_activations():
    dt = 1/30
    dur = 20
    ts = np.arange(0, dur, dt)
    
    tau0 = 3.0
    speed = 20.0
    dist = tau0*speed - ts*speed
    tau = dist/speed

    param = dict(
            dt=dt,
            std=0.75,
            damping=1.6,
            tau_threshold=2.3,
            scale=1.0,
            act_threshold=1.0,
            pass_threshold=0.0
    )
    N = 100
    act, crossing_prob, actgrid = vdd_activation_pdf(VddmParams(**param), tau, N=N)
    #crossing_prob = vdd_decision_pdf(hacktau, dt, N=500, minact=-10, **param)
    #plt.plot(actgrid, act[30])
    #plt.show()
    dead = np.cumsum(crossing_prob*dt)
    alive = 1 - dead
    plt.pcolormesh(ts, actgrid, np.sqrt(act*alive.reshape(-1, 1) + np.finfo(float).eps).T, cmap='plasma', antialiased=True)
    plt.colorbar()
    plt.twinx()
    plt.plot(ts, crossing_prob)
    plt.show()

def test_blocked():
    dt = 1/30
    dur = 20
    ts = np.arange(0, dur, dt)
    
    tau0 = 8.0
    speed = 20.0
    dist = tau0*speed - ts*speed
    tau = dist/speed
    
    tau0b = tau0 - 3.0

    distb = tau0b*speed - ts*speed
    taub = distb/speed
    
    param = dict(
            dt=dt,
            std=0.75,
            damping=1.6,
            tau_threshold=2.3,
            scale=1.0,
            act_threshold=1.0,
            pass_threshold=0.0
    )
    
    (act, actb), (crossing_prob, unblock_prob), actgrid = vdd_blocker_activation_pdf(
            VddmParams(**param), tau, taub, N=200,
            #minact=-1.0, maxact=1.0
            )
    #plt.plot(ts, crossing_prob)
    #plt.plot(ts, 1 - np.cumsum(unblock_prob))
    
    uncrossed = 1.0 - np.cumsum(crossing_prob*dt)
    act_dens = act*uncrossed.reshape(-1, 1)
    a = act
    plt.pcolormesh(ts, actgrid, a.T, vmax=np.max(a[10:]), cmap='jet')
    plt.twinx()
    plt.plot(ts, crossing_prob)
    plt.show()

def test_gridsize():
    dt = 1/30
    dur = 20
    ts = np.arange(0, dur, dt)
    
    tau0 = 3.0
    speed = 20.0
    dist = tau0*speed - ts*speed
    tau = dist/speed
    
    param = dict(
            dt=dt,
            std=0.75,
            damping=1.6,
            tau_threshold=2.3,
            scale=1.0,
            act_threshold=1.0,
            pass_threshold=0.0
    )
    
    for N in [25, 50, 100, 200]:
        crossing_prob = vdd_decision_pdf(VddmParams(**param), tau, N=N)
        plt.plot(ts, crossing_prob, label=N)
    plt.legend()
    plt.show()
    



def benchmark():
    dt = 1/30
    dur = 10
    ts = np.arange(0, dur, dt)
    
    tau0 = 3.0
    speed = 20.0
    dist = tau0*speed - ts*speed
    tau = dist/speed
    
    tau0b = tau0 - 0.0

    distb = tau0b*speed - ts*speed
    taub = distb/speed
    
    param = dict(
            dt=dt,
            std=0.5,
            damping=0.5,
            tau_threshold=2.5,
            scale=1.0,
            act_threshold=0.5,
            pass_threshold=0.0
    )
    N = 100

    # Compilation
    vdd_decision_pdf(VddmParams(**param), tau, N=N)
    import time
    trials = 10
    st = time.perf_counter()
    for i in range(trials):
        pdf = vdd_decision_pdf(VddmParams(**param), tau, N=N)
    dur = time.perf_counter() - st
    print(f"{dur/trials*1000} ms per run")

    plt.plot(ts, pdf)
    plt.show()




if __name__ == '__main__':
    #test_activations()
    #test_blocked()
    #benchmark()
    test_gridsize()

