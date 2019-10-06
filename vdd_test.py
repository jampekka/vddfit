import numpy as np
import matplotlib.pyplot as plt
import numba
from pprint import pprint
import time

@numba.njit(fastmath=True)
def vdd(y, x0, alpha, noise):
    x = x0
    xs = np.empty_like(y)
    for i in range(len(xs)):
        x = alpha*x + y[i] + noise[i]
        xs[i] = x
    return xs

@numba.njit(fastmath=True)
def vdd_time(y, x0, alpha, noise, threshold):
    act = vdd(y, x0, alpha, noise)
    if x0 > threshold:
        return 0
    prev = x0
    for i in range(len(act)):
        if act[i] <= threshold:
            prev = act[i]
            continue
        t = (threshold - prev)/(act[i] - prev)
        return i + t
        #return max(0.0, i + t)

    return np.nan

@numba.njit(fastmath=True)
def vdd_times(y, x0, alpha, noises, threshold):
    out = np.empty(noises.shape[0])
    for i in range(len(noises)):
        out[i] = vdd_time(y, x0, alpha, noises[i], threshold)
    return out

def timeit(func, n=10):
    times = []
    for i in range(n):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return times

dt = 0.01
dur = 20
ts = np.arange(0, dur, dt)

tau0 = 4.0
speed = 30.0

dist = tau0*speed - ts*speed
tau = dist/speed

threshold = 4
scale = 0.5
tau[tau < 0] = 4.5
evidence = np.arctan((tau - threshold)/scale)
#plt.plot(ts, tau)
#plt.plot(ts, evidence + threshold)
#plt.show()

N = 10000
noises = np.random.randn(N, len(ts))*(3.0*np.sqrt(dt))

times = vdd_times(evidence, 0.0, 0.9, noises, 1.0)

print(timeit(lambda: vdd_times(evidence, 0.0, 0.9, noises, 1.0)))

plt.hist(times[np.isfinite(times)]*dt, bins=100, density=True)

plt.show()

for i in range(3):
    act = vdd(evidence, 0, 0.9, noises[i])
    #plt.plot(act)
    plt.plot(ts, act, alpha=0.1)
plt.plot(ts, evidence)
#plt.plot(ts, tau)
plt.show()
