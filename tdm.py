import numpy as np
import scipy.stats
from scipy.interpolate import interp1d

tdistr = lambda a, b: scipy.stats.lognorm(a, scale=b)
rtdistr = lambda a, b: scipy.stats.lognorm(a, scale=b)

def cross_time_dist(dt, taus, tdist, rtdist, passed_ttc=0.0):
    ts = np.arange(len(taus))*dt

    ttcs = taus.copy()
    ttcs[~np.isfinite(ttcs)] = np.inf
    ttcs[ttcs < passed_ttc] = np.inf
    maxs = np.maximum.accumulate(ttcs)
    done = tdist.cdf(maxs)
    #done = dist.sf(timer(ts))
    try:
        dts = np.ediff1d(done, to_begin=float(done[0]))/dt
    except ValueError as e:
        # This happens sometimes, so let's just propagate the
        # nans and hope the optimizer deals with it
        if np.all(~np.isfinite(done)):
            dts = done.copy()
        else:
            raise
    latency = np.ediff1d(rtdist.cdf(ts), to_end=[0.0])/dt
    latency = [0.0]*(len(latency) - 1) + list(latency)
    dts = np.convolve(dts, np.array(latency), mode='valid')
    #dts = dts*(1 - slack) + slack
    dts *= dt
    nonresp = 1 - done[-1]
    interp = interp1d(ts, dts, fill_value=(np.nan, nonresp), bounds_error=False)
    return scipy.interpolate.interp1d(ts, dts)

def predict_decision_times(dt, taus, tm=0.6, ts=3.3, rtm=0.8, rts=1.0, *args, **kwargs):
    tdist = tdistr(tm, ts)
    rtdist = rtdistr(rtm, rts)

    return cross_time_dist(dt, taus, tdist, rtdist, *args, **kwargs)
