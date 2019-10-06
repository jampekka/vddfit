from numba import njit, jit, prange
import numpy as np
import matplotlib.pyplot as plt

@jit(nopython=True, parallel=True)
def simulate_activation(tau, dt, noise, std, damping, scale, tau_threshold, *args):
    act = 0.0
    for i in range(len(tau)):
        act += (-damping*act + np.arctan((tau[i] - tau_threshold)*scale) + noise[i]*std)*dt
        yield act

#@njit
@jit(nopython=True, parallel=True)
def simulate_time(tau, dt, noise, std, damping, scale, tau_threshold, act_threshold):
    acts = simulate_activation(tau, dt, noise, std, damping, scale, tau_threshold)
    prev = 0.0
    if prev > act_threshold:
        return 0.0

    for i, act in enumerate(acts):
        if act < act_threshold:
            prev = act
            continue
        
        t = (act_threshold - prev)/(act - prev)
        return (i + t)*dt
    return np.nan

@jit(nopython=True, parallel=True)
def simulate_times(tau, dt, noise_bank, std, damping, scale, tau_threshold, act_threshold):
    out = np.empty(len(noise_bank))
    for i in prange(len(noise_bank)):
        out[i] = simulate_time(tau, dt, noise_bank[i], std, damping, scale, tau_threshold, act_threshold)
    return out

@njit
def stdnormpdf(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

@njit
def sample_lik(vals, sample, dt):
    liks = np.empty_like(vals)
    # TODO: Handle explicitly
    sample = sample[np.isfinite(sample)]
    n = len(sample)
    std = 5*np.sqrt(dt) # TODO: Find a principled value for this?
    bw = std*n**(-1/(1+4))
    for i, val in enumerate(vals):
        lik = 0.0
        for s in sample:
            # TODO: Make a discrete PDF from the empirical CDF?
            lik += stdnormpdf((s - val)/bw)
        liks[i] = lik/(n*bw)
    return liks

from kwopt import minimizer, logbarrier, logitbarrier

def vdd_loss(trials, dt, N=5000):
    taus, rts = zip(*trials)
    hacktaus = []
    for tau in taus:
        hacktau = tau.copy()
        hacktau[hacktau < 0] = 1e5
        hacktaus.append(hacktau)
    noises = [np.random.randn(N, len(tau)) for (tau, rts) in trials]

    def loss(**kwargs):
        lik = 0
        for tau, rt, noise in zip(hacktaus, rts, noises):
            sample = simulate_times(tau, dt, noise, **kwargs)
            lik += np.sum(np.log(sample_lik(rt, sample, dt) + 1e-9))

        return -lik
    
    return loss


def fit_vdd(trials, dt, N=1000, init=None):
    if init is None:
        init = dict(
            std=1.0*np.sqrt(dt),
            damping=0.5,
            scale=1.0,
            tau_threshold=3.5,
            act_threshold=1.0
            )
    spec = dict(
        std=            (init['std'], logbarrier),
        damping=      (init['damping'], logitbarrier),
        scale=          (init['scale'], logbarrier),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier)
            )
    
    loss = vdd_loss(trials, dt, N)
    return minimizer(loss, method='nelder-mead')(**spec)

def gridtest():
    N = 20
    dt = 1/30
    dur = 20
    ts = np.arange(0, dur, dt)
    param = dict(
        std=1.0,
        damping=0.5,
        scale=1.0,
        tau_threshold=3.5,
        act_threshold=1.0
        )

    trials = []
    for tau0 in (2.0, 3.0, 4.0, 5.0):
        speed = 20.0
        dist = tau0*speed - ts*speed
        tau = dist/speed

        np.random.seed(0)
        noise_bank = np.random.randn(N, len(tau))

        hacktau = tau.copy()
        hacktau[hacktau < 0] = 1e5

        sample = simulate_times(hacktau, dt, noise_bank, **param)
        trials.append((tau, sample))
    
    #np.random.seed(0)
    #noise_bank = np.random.randn(N, len(tau))


    
    loss = vdd_loss(trials, dt)
    liks = []
    
    stds = np.linspace(0.1, 3, 30)
    #for std in stds:
    #    liks.append(-loss(**{**param, **{'std': std}}))
    #plt.plot(stds/np.sqrt(dt), liks)
    
    thresholds = np.linspace(2.0, 6.0, 30)

    S, T = np.meshgrid(stds, thresholds)
    for std, threshold in zip(*(x.flat for x in (S, T))):
        liks.append(-loss(**{**param, **{'tau_threshold': threshold, 'std': std}}))
    
    liks = np.array(liks)
    plt.pcolormesh(S, T, np.exp(liks.reshape(S.shape)))
    plt.plot(param['std'], param['tau_threshold'], 'ro')
    plt.colorbar()
    #for threshold in thresholds:
    #    liks.append(-loss(**{**param, **{'tau_threshold': threshold}}))
    #plt.plot(thresholds, np.exp(liks))
    
    plt.show()


def fittingtest():
    N = 20
    dt = 1/90
    dur = 20
    ts = np.arange(0, dur, dt)
    param = dict(
        std=1.0,
        damping=0.5,
        scale=1.0,
        tau_threshold=2.0,
        act_threshold=1.0
        )

    trials = []
    for tau0 in (2.0, 3.0, 4.0, 5.0):
        tau0 = 4.5
        speed = 30.0
        dist = tau0*speed - ts*speed
        tau = dist/speed

        np.random.seed(0)
        noise_bank = np.random.randn(N, len(tau))

        hacktau = tau.copy()
        hacktau[hacktau < 0] = 1e5

        sample = simulate_times(hacktau, dt, noise_bank, **param)
        trials.append((tau, sample))

    result = fit_vdd(trials, dt)
    print(result)
    #plt.hist(sample)
    #plt.show()


def samplingtest():
    N = 5000

    dt = 1/30
    dur = 20
    ts = np.arange(0, dur, dt)

    tau0 = 5.0
    speed = 30.0
    dist = tau0*speed - ts*speed
    tau = dist/speed

    np.random.seed(0)
    noise_bank = np.random.randn(N, len(tau))

    tau[tau < 0] = 1e5

    param = dict(
        std=3.0,
        damping=2.5,
        scale=1.0,
        tau_threshold=4.0,
        act_threshold=1.0
        )

    #print("Simulating")
    sample = simulate_times(tau, dt, noise_bank, **param)

    #responders = np.isfinite(sample)
    #print(np.sum(responders)/len(responders)*100)
    #plt.hist(sample[responders]*dt, bins=100)
    for i in range(10):
        act = np.array(list(simulate_activation(tau, dt, noise_bank[i], std=param['std'], damping=param['damping'], scale=param['scale'], tau_threshold=param['tau_threshold'])))
        plt.plot(ts, act)
    plt.figure()

    plt.hist(sample[np.isfinite(sample)], bins=100, density=True)
    est = sample_lik(ts, sample, dt)
    #from scipy.stats.kde import gaussian_kde
    #est = gaussian_kde(sample, bw_method=0.1)(ts)
    #plt.plot(ts, est, color='green')
    #plt.twinx()
    print(sample)
    plt.plot(ts, est, color='red')

    plt.show()

if __name__ == '__main__':
    #gridtest()
    #fittingtest()
    samplingtest()
