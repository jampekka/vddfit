import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit, njit, prange, vectorize, float64, jitclass
import numba as nb
from collections import namedtuple
import scipy.interpolate
#def njit(*args, **kwargs):
#    return lambda f: f

# TODO: Rewrite in cython

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

@jitclass(dict(start=float64, dv=float64, N=nb.uint64))
class Grid1d:
    def __init__(self, start, end, N):
        self.start = start
        self.dv = (end - start)/N
        self.N = N

    def __getitem__(self, i):
        return self.start + i*self.dv
    
    @property
    def size(self):
        return self.N

    @property
    def end(self):
        return self.start + self.N*self.dv

    def closest(self, x):
        return int((x - self.start)/self.dv)
    
    @property
    def values(self):
        return np.linspace(self.start, self.end, self.N)


@jitclass(dict(grid=Grid1d.class_type.instance_type, y=float64[:], upper_fill=float64))
class GridInterp1d:
    def __init__(self, grid, y, upper_fill):
        self.grid = grid
        self.y = y
        self.upper_fill = upper_fill

    def get(self, x):
        if x < self.grid.start: return np.nan
        if x > self.grid.end: return self.upper_fill
        return self.y[self.grid.closest(x)] # TODO: linear interpolation


@njit(parallel=False, fastmath=True)
def vdd_step(p, da, tau, acts, pweights, nweights, decision_prob=1.0):
    decided = 0.0
    N = acts.size
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
            #diff = acts[to] - act_fr
            small_enough = stdnormcdf((acts[to] - act_fr + da/2 - diff_mean)/hacks)
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
    #acts = np.linspace(minact, maxact, N)
    #da = acts[1] - acts[0]
    acts = Grid1d(minact, maxact, N)
    da = acts.dv

    pweights = np.zeros(N)
    #pweights[np.searchsorted(acts, 0.0)] = 1.0
    pweights[acts.closest(0.0)] = 1.0
    nweights = np.zeros_like(pweights)
    
    deadpdf = np.empty(len(taus))
    alive = 1.0

    for i, tau in enumerate(taus):
        dead = alive*vdd_step(p, da, tau, acts, pweights, nweights)
        pweights, nweights = nweights, pweights
        nweights[:] = 0

        alive -= dead
        deadpdf[i] = dead
        
    return GridInterp1d(Grid1d(0.0, len(taus)*p.dt, len(taus)), crossedpdf/p.dt, uncrossed)

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
    
    #cp = p._replace(pass_threshold=0.0)
    cp = VddmParams(
        dt=p.dt, std=p.std, damping=p.damping, tau_threshold=p.tau_threshold, pass_threshold=0.0,
        scale=p.scale, act_threshold=p.act_threshold
    )
    for i, (tau, btau) in enumerate(zip(taus, blocker_taus)):
        gone = 1.0 if btau <= p.pass_threshold else 0.0
        unblocked = blocked*vdd_step(p, da, btau, acts, bweights[i], bweights[i+1], decision_prob=gone)
        blocked -= unblocked
        
        crossed = uncrossed*vdd_step(cp, da, tau, acts, weights[i], weights[i+1], 1.0 - blocked)
        uncrossed -= crossed
        
        crossedpdf[i] = crossed
        unblockedpdf[i] = unblocked
        
    return (weights[1:], bweights[1:]), (crossedpdf/p.dt, unblockedpdf/p.dt), acts

@njit
def vdd_blocker_decision_pdf(p, taus, blocker_taus, N=50, minact=DEFAULT_MINACT, maxact=DEFAULT_MAXACT):
    acts = Grid1d(minact, maxact, N)
    da = acts.dv
    #acts = np.linspace(minact, maxact, N)
    #da = acts[1] - acts[0]
    
    weights = np.zeros(N)
    pweights = np.zeros(N)
    pweights[acts.closest(0.0)] = 1.0
    #pweights[np.searchsorted(acts, 0.0)] = 1.0
    
    weights_b = np.zeros(N)
    pweights_b = np.zeros(N)
    pweights_b[acts.closest(0.0)] = 1.0
    #pweights_b[np.searchsorted(acts, 0.0)] = 1.0
    
    
    crossedpdf = np.empty(len(taus))
    unblockedpdf = np.empty(len(taus))
    uncrossed = 1.0
    blocked = 1.0
    
    #cp = p._replace(pass_threshold=0.0)
    # Damn numba!
    cp = VddmParams(
        dt=p.dt, std=p.std, damping=p.damping, tau_threshold=p.tau_threshold, pass_threshold=0.0,
        scale=p.scale, act_threshold=p.act_threshold
    )
    for i, (tau, btau) in enumerate(zip(taus, blocker_taus)):
        gone = 1.0 if btau <= p.pass_threshold else 0.0
        unblocked = blocked*vdd_step(p, da, btau, acts, pweights_b, weights_b, decision_prob=gone)
        blocked -= unblocked
        weights_b, pweights_b = pweights_b, weights_b
        weights_b[:] = 0
        
        crossed = uncrossed*vdd_step(cp, da, tau, acts, pweights, weights, 1.0 - blocked)
        uncrossed -= crossed
        weights, pweights = pweights, weights
        weights[:] = 0
        
        crossedpdf[i] = crossed
        unblockedpdf[i] = unblocked
    
    return GridInterp1d(Grid1d(0.0, len(taus)*p.dt, len(taus)), crossedpdf/p.dt, uncrossed)
    #return scipy.interpolate.interp1d(
    #        np.linspace(0, len(taus)*p.dt, len(taus)), crossedpdf/p.dt,
    #        fill_value=(np.nan, uncrossed),
    #        bounds_error=False
    #        )

def vdd_loss(trials, dt, N=100):
    taus, rts = zip(*trials)
    ts = [np.arange(len(tau))*dt for tau in taus]

    def loss(**kwargs):
        lik = 0.0
        for tau, rts in zip(taus, rts):
            kwargs['dt'] = dt
            pdf = vdd_decision_pdf(VddmParams(**kwargs), tau, N=N)
            #lik += np.sum(np.log(pdf(rt) + eps))
            for rt in rts: lik += np.log(pdf.get(rt) + eps)

        return -lik
    
    return loss

def vdd_blocker_loss(trials, dt, N=100):
    taus, btaus, rtss = zip(*trials)
    def loss(p):
        lik = 0.0
        for i in range(len(taus)):
            tau = taus[i]
            btau = btaus[i]
            rts = rtss[i]
            pdf = vdd_blocker_decision_pdf(p, tau, btau, N=N)
            for rt in rts: lik += np.log(pdf.get(rt) + eps)

        return -lik
    
    return lambda **kwargs: loss(VddmParams(**dict(dt=dt, **kwargs)))

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
    plt.pcolormesh(ts, actgrid, np.sqrt(act*alive.reshape(-1, 1) + np.finfo(float).eps).T, cmap='jet', antialiased=True)
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

    plt.figure()
    plt.pcolormesh(ts, actgrid, actb.T, vmax=np.max(a[10:]), cmap='jet')
    plt.twinx()
    plt.plot(ts, unblock_prob)
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

    plt.plot(ts, pdf.y)
    plt.show()




if __name__ == '__main__':
    #test_activations()
    test_blocked()
    #benchmark()
    #test_gridsize()

