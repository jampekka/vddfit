import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pprint import pprint
import functools

# TODO: LRU memoization
# TODO: Loss function
# TODO: Fit to actual data

@njit
def linear_activation(evidence, noise, smoothing, x0=0.0, out=None):
    x = x0
    if out is None:
        out = np.empty_like(evidence)
    for i in range(len(evidence)):
        x = smoothing*x + evidence[i] + noise[i]
        out[i] = x
    return out

@njit
def linear_activations(evidence, noises, smoothing, x0=0.0, out=None):
    x = x0
    if out is None:
        out = np.empty(len(noises), len(evidence))
    for i in range(len(noises)):
        linear_activation(evidence, noises[i], smoothing, x0, out[i])
    return out

@njit
def threshold_time(act, threshold, x0=0.0):
    prev = x0
    for i in range(len(act)):
        if act[i] <= threshold:
            prev = act[i]
            continue
        t = (threshold - prev)/(act[i] - prev)
        return i + t
        #return max(0.0, i + t)
    return np.nan

@njit
def threshold_times(acts, threshold, x0=0.0):
    prev = x0
    out = np.empty(len(acts))
    for i in range(len(acts)):
        out[i] = threshold_time(acts[i], threshold, x0)
    return out

def sample_loglik(sample, val):
    # TODO: Proper normalization!
    loglik = 0.0
    for particle in sample:
        loglik += (sample - val)**2
    return loglik


def rt_distr(tau, noise_bank):
    def get_activations(std, scale, smoothing, tau_threshold):
        evidence = np.arctan((tau - tau_threshold)/scale)
        noises = noise_bank*std*np.sqrt(dt)
        return linear_activations(evidence, noises, smoothing)

    def distr(std, scale, smoothing, tau_threshold, act_threshold):
        activations = get_activations(std, scale, smoothing, tau_threshold)
        samples = threshold_times(activations, act_threshold)
        return lambda v: sample_loglik(samples, v)

    return distr
        


N = 10000
dt = 1/90 # Check!

"""
def gridit(trajectories, noise_bank):
    stds = np.linspace(0.01, 5.0, 10)
    scales = np.linspace(0.01, 10.0, 10)
    smoothings = np.linspace(0.0, 1.0, 10)
    act_thresholds = np.linspace(0.01, 10.0, 10)
    tau_thresholds = np.linspace(1.0, 10.0, 10)

    for tau, reaction_times in trajectories:
        for std in stds:
            noises = noise_bank*std*np.sqrt(dt)
            for scale in scales:
                for tau_threshold in tau_thresholds:
                    evidence = np.arctan((tau - tau_threshold)/scale)
                    for smoothing in smoothings:
                        acts = []
                        for i in range(len(noises)):
                            act = linear_activations(evidence, noises[i], smoothing)
                            acts.append(act)

                        for act_threshold in act_thresholds:
                            rts = []
                            for act in acts:
                                rt = threshold_time(act, act_threshold)
                                rts.append(rt)
                            rts = np.array(rts)
                            params = dict(
                                    std=std,
                                    scale=scale,
                                    tau_threshold=tau_threshold,
                                    smoothing=smoothing,
                                    act_threshold=act_threshold)
                            pprint(params)

                            plt.hist(rts[np.isfinite(rts)], density=True)
                            plt.show()
"""
dur = 20
ts = np.arange(0, dur, dt)

tau0 = 4.0
speed = 30.0

dist = tau0*speed - ts*speed
tau = dist/speed
np.random.seed(0)
noise_bank = np.random.randn(N, len(tau))

#trajectories = [
#        (tau, [])
#        ]
#gridit(trajectories, noise_bank)
distr = rt_distr(tau, noise_bank)
dist = distr(
        std=1.0*np.sqrt(dt),
        smoothing=0.5,
        scale=1.0,
        tau_threshold=4.0,
        act_threshold=1.0
        )
plt.plot(ts, dist(ts))
plt.show()
