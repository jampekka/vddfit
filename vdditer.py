from numba import njit
import numpy as np
import matplotlib.pyplot as plt

@njit
def simulate_activation(tau, noise, std, smoothing, scale, tau_threshold, *args):
    act = 0.0
    for i in range(len(tau)):
        act = act*smoothing + np.arctan((tau[i] - tau_threshold)*scale) + noise[i]*std
        yield act

@njit
def simulate_time(tau, noise, std, smoothing, scale, tau_threshold, act_threshold):
    acts = simulate_activation(tau, noise, std, smoothing, scale, tau_threshold)
    prev = 0.0
    if prev > act_threshold:
        return 0.0

    for i, act in enumerate(acts):
        if act < act_threshold:
            prev = act
            continue
        
        t = (act_threshold - prev)/(act - prev)
        return i + t
    return np.nan

@njit
def simulate_times(tau, noise_bank, std, smoothing, scale, tau_threshold, act_threshold):
    out = np.empty(len(noise_bank))
    for i in range(len(noise_bank)):
        out[i] = simulate_time(tau, noise_bank[i], std, smoothing, scale, tau_threshold, act_threshold)
    return out

@njit
def stdnormpdf(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

@njit
def sample_lik(vals, sample, dt):
    liks = np.empty_like(vals)
    n = len(sample)
    std = 0.1
    bw = std*n**(-1/(1+4))
    for i, val in enumerate(vals):
        lik = 0.0
        for s in sample:
            # TODO: Handle non-responders
            # TODO: Make a discrete PDF from the empirical CDF?
            lik += stdnormpdf((s*dt - val)/bw)
        liks[i] = lik/(n*bw)
    return liks

from kwopt import minimizer, logbarrier, logitbarrier

def vdd_loss(trials, dt, N=1000):
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
            sample = simulate_times(tau, noise, **kwargs)
            lik += np.sum(np.log(sample_lik(rt, sample, dt)))

        return -lik
    
    return loss


def fit_vdd(trials, dt, N=1000, init=None):
    if init is None:
        init = dict(
            std=1.0*np.sqrt(dt),
            smoothing=0.5,
            scale=1.0,
            tau_threshold=4.0,
            act_threshold=1.0
            )
    spec = dict(
        std=            (init['std'], logbarrier),
        smoothing=      (init['smoothing'], logitbarrier),
        scale=          (init['scale'], logbarrier),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier)
            )
    
    loss = vdd_loss(trials, dt, N)
    return minimizer(loss, method='nelder-mead')(**spec)

def gridtest():


def fittingtest():
    N = 20
    dt = 1/90
    dur = 20
    ts = np.arange(0, dur, dt)
    param = dict(
        std=1.0*np.sqrt(dt),
        smoothing=0.5,
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

        sample = simulate_times(hacktau, noise_bank, **param)*dt
        trials.append((tau, sample))

    result = fit_vdd(trials, dt)
    print(result)
    #plt.hist(sample)
    #plt.show()


def samplingtest():
    N = 1000

    dt = 1/90
    dur = 20
    ts = np.arange(0, dur, dt)

    tau0 = 4.5
    speed = 30.0
    dist = tau0*speed - ts*speed
    tau = dist/speed

    np.random.seed(0)
    noise_bank = np.random.randn(N, len(tau))

    tau[tau < 0] = 4.5

    param = dict(
        std=1.0*np.sqrt(dt),
        smoothing=0.5,
        scale=1.0,
        tau_threshold=4.0,
        act_threshold=1.0
        )

    #print("Simulating")
    sample = simulate_times(tau, noise_bank, **param)

    #responders = np.isfinite(sample)
    #print(np.sum(responders)/len(responders)*100)
    #plt.hist(sample[responders]*dt, bins=100)

    plt.hist(sample*dt, bins=100, density=True)
    est = sample_lik(ts, sample, dt)
    #from scipy.stats.kde import gaussian_kde
    #est = gaussian_kde(sample, bw_method=0.1)(ts)
    #plt.plot(ts, est, color='green')
    #plt.twinx()
    plt.plot(ts, est, color='red')

    plt.show()

if __name__ == '__main__':
    fittingtest()
    #samplingtest()
