import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit, njit
from collections import namedtuple
#def njit(*args, **kwargs):
#    return lambda f: f

VddmParams = namedtuple('VddmParams',
    'dt     std     damping     tau_threshold   pass_threshold  scale   act_threshold',
    defaults=(                                  0.0,            1.0,    1.0)
        )

eps = np.finfo(float).eps

@njit()
def normpdf(x, m, v):
    return np.exp(-(m - x)**2/(2*v))/np.sqrt(2*np.pi*v)

# TODO: The cdf and sf computations should be numerically better!
@njit()
def normcdf(x, m, v):
    return (1.0 + math.erf((x - m)/np.sqrt(v*2)))/2.0

@njit()
def normsf(*args):
    return 1.0 - normcdf(*args)

@njit(parallel=True)
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
    
    for i in range(N):
        act = acts[i]
        diff_mean = diff_mean_tau - alpha*act
        
        for j in range(N):
            diff = acts[j] - act
            
            # TODO: The probabilities don't sum to one on small da. May be numerical
            #   problems, but may be buggy code. Check especially the boundary stuff!
            # TODO: This computes every edge cdf twice (IN THE MOST INNER LOOP!!),
            #  whereas one would suffice and the boundaries would be handled nicer.
            if j == 0:
                trans_prob = normcdf(diff + da/2, diff_mean,  diff_var)
            elif j == N - 1:
                trans_prob = 1.0 - normcdf(diff - da/2, diff_mean, diff_var)
            else:
                # act + N(dm, dv) < acts[j] + da/2
                # N(dm, dv) < acts[j] - act + da/2
                smallenough = normcdf(diff + da/2, diff_mean,  diff_var)
                # act + N(dm, dv) > acts[j] - da/2
                # N(dm, dv) > acts[j] - act - da/2
                bigenough = 1.0 - normcdf(diff - da/2, diff_mean, diff_var)
                trans_prob = bigenough*smallenough

            nweights[j] += trans_prob*pweights[i]
    
    # Doing the decisions after normalizing due to the numerical(?) issues
    nweights /= np.sum(nweights)
    for i in range(N):
        a = acts[i]
        if a < p.act_threshold - da/2: continue # TODO: Optimize
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
def vdd_activation_pdf(p, taus, N=100, minact=-5.0, maxact=5.0):
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
def vdd_decision_pdf(p, taus, N=100, minact=-5.0, maxact=5.0):
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
def vdd_blocker_activation_pdf(p, taus, blocker_taus, N=100, minact=-5.0, maxact=5.0):
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
    hacktaus = []
    for tau in taus:
        hacktau = tau.copy()
        hacktau[hacktau < 0] = 1e5
        hacktaus.append(hacktau)

    def loss(**kwargs):
        lik = 0
        for tau, rti in zip(hacktaus, rtis):
            #pdf = vdd_decision_pdf(tau, dt, **kwargs, N=N)
            kwargs['dt'] = dt
            pdf = vdd_decision_pdf(VddmParams(**kwargs), tau, N=N)
            lik += np.sum(np.log(pdf[rti] + eps))

        return -lik
    
    return loss

def test():
    dt = 1/30
    dur = 10
    ts = np.arange(0, dur, dt)
    
    tau0 = 3.0
    speed = 20.0
    dist = tau0*speed - ts*speed
    tau = dist/speed

    hacktau = tau.copy()
    #hacktau[hacktau < 0] = 1e5
    
    param = dict(
            dt=dt,
            std=0.5,
            damping=0.5,
            tau_threshold=2.5,
            scale=1.0,
            act_threshold=0.5,
            pass_threshold=0.0
    )
    #for std in np.linspace(0.1, 2.0, 1):
    for N in (50,):
        #pdf = vdd_pdf(hacktau, dt, **param)
        act, crossing_prob, actgrid = vdd_activation_pdf(VddmParams(**param), hacktau, N=N)
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
    
    (act, actb), (crossing_prob, unblock_prob), actgrid = vdd_blocker_activation_pdf(VddmParams(**param), tau, taub, N=200)
    plt.plot(ts, crossing_prob)
    plt.plot(ts, unblock_prob)

    plt.show()

if __name__ == '__main__':
    test_blocked()

