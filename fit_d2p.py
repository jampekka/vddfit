import pandas as pd
import numpy as np
from vdd_disc import vdd_loss, vdd_decision_pdf, VddmParams
import tdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize
from pprint import pprint

all_trajectories = pd.read_csv('d2p_trajectories.csv')
all_responses = pd.read_csv('d2p_cross_times_uk.csv')
all_responses[["subject_id", "trial_number"]] = all_responses.unique_ID.str.split("_", 1, expand=True)

responses = dict(list(all_responses.groupby('trial_id')))

trials = {}
trial_taus = {}
subj_trial_responses = {}

for trial, traj in all_trajectories.groupby('trial_n'):
    if np.std(traj.speed) > 0.01:
        continue
    resp = responses[trial]['cross_time'].values
    #tau = traj['tau'].values
    tau = traj['distance'].values/traj['speed'].values
    tau[tau < 0] = 1e50
    trials[trial] = (tau, resp)
    trial_taus[trial] = tau

for s, sd in all_responses.groupby('subject_id'):
    st_rts = subj_trial_responses[s] = {}
    for t, td in sd.groupby('trial_id'):
        if t not in trial_taus: continue
        st_rts[t] = td.cross_time.values

#dt = np.median(np.diff(all_trajectories['time_c']))
dt = 1/30

param = dict(
       std=0.9,
       damping=0.4,
       scale=1.0,
       tau_threshold=4.3,
       act_threshold=1.0
       )
#param = {'std': 0.6970908298337709, 'damping': 0.46041074483528815, 'scale': 1.0, 'tau_threshold': 2.0542953634220624, 'act_threshold': 1.0}
#param = {'std': 0.6944196353260089, 'damping': 0.4747255460581086, 'scale': 0.7848596401338991, 'tau_threshold': 1.984151857035362, 'act_threshold': 1.0}

loss = vdd_loss(list(trials.values()), dt, N=10)
liks = []

from kwopt import minimizer, logbarrier, logitbarrier, fixed
def fit_vdd(trials, dt, init=None):
    if init is None:
        init = param.copy()

    spec = dict(
        std=            (init['std'], logbarrier),
        damping=      (init['damping'], logbarrier),
        scale=          (init['scale'], logbarrier, fixed),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier, fixed)
            )
    
    loss = vdd_loss(trials, dt)
    def cb(x, f, accept):
        print(f, accept, x)
    return minimizer(loss, method='powell')(**spec)

def fit_tdm(trials, dt):
    spec = dict(
        tm=             (1.0, logbarrier),
        ts=             (1.0, logbarrier),
        rtm=            (1.0, logbarrier),
        rts=            (1.0, logbarrier),
            )
    
    def loss(**kwargs):
        lik = 0
        for taus, rts in trials:
            pdf = tdm.predict_decision_times(dt, taus, **kwargs)
            lik += np.sum(np.log(pdf(rts) + 1e-9))
        return -lik
    return minimizer(loss, method='powell')(**spec)




def singlefit():
    fit = fit_vdd(list(trials.values()), dt)
    print(fit)
    tdmfit = fit_tdm(list(trials.values()), dt)
    print(tdmfit)
    #param = fit.kwargs

    #griddens = 100
    #stds = np.linspace(0.8, 1.2, griddens)
    #thresholds = np.linspace(3.0, 5.0, griddens)
    #for std in stds:
    #    liks.append(-loss(**{**param, **{'std': std}}))
    #plt.plot(stds, liks)
    #for th in thresholds:
    #    liks.append(-loss(**{**param, **{'tau_threshold': th}}))
    #plt.plot(thresholds, liks)

    """
    S, T = np.meshgrid(stds, thresholds)
    N = np.product(S.shape)
    for i, (std, threshold) in enumerate(zip(*(x.flat for x in (S, T)))):
        print(f"{i/N*100}% done")
        liks.append(-loss(**{**param, **{'tau_threshold': threshold, 'std': std}}))

    liks = np.array(liks)
    winner = np.nanargmax(liks)
    std, threshold = S.flat[winner], T.flat[winner]
    param = {**param, **{'tau_threshold': threshold, 'std': std}}
    """

    pdf = PdfPages("gradfit.pdf")
    def show():
        pdf.savefig()
        plt.close()

    """
    plt.title("Parameterization likelihood")
    plt.pcolormesh(S, T, np.exp(liks.reshape(S.shape)))
    plt.plot(std, threshold, 'ro', label='Maximum likelihood')
    plt.xlabel("Noise std")
    plt.ylabel("Tau threshold")
    plt.colorbar()
    plt.legend()
    show()
    """

    for trial, (tau, resp) in trials.items():
        ts = np.arange(len(tau))*dt
        plt.title(f"Trial type {trial}")
        plt.hist(resp, bins=20, density=True, label='Measurements')
        #plt.hist(sample[np.isfinite(sample)], bins=100, histtype='step', density=True)
        kwargs = fit.kwargs
        kwargs['dt'] = dt
        vdd = vdd_decision_pdf(VddmParams(**kwargs), tau)
        plt.plot(ts, vdd, label='VDDM')

        tdm_pdf = tdm.predict_decision_times(dt, tau, **tdmfit.kwargs)(ts)
        plt.plot(ts, tdm_pdf, label='TDM')
        plt.ylabel("Crossing likelihood")
        plt.legend()
        plt.twinx()
        plt.plot(ts, tau, color='black', label='Tau')
        plt.ylabel("Tau")
        plt.xlabel("Time")
        plt.ylim(0, 10)
        show()

    pdf.close()

def fit_vdd_eb(trials, dt, init=None, prior=lambda **params: 1):
    if init is None:
        init = param.copy()
    
    spec = dict(
        std=            (init['std'], logbarrier),
        damping=      (init['damping'], logbarrier),
        scale=          (init['scale'], logbarrier, fixed),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier, fixed)
            )
    
    raw_loss = vdd_loss(trials, dt)
    loss = lambda **params: prior(**params)*raw_loss(**params)
    def cb(x, f, accept):
        print(f, accept, x)
    #return minimizer(loss, scipy.optimize.basinhopping, callback=cb, minimizer_kwargs=dict(method='powell'))(**spec)
    return minimizer(loss, method='powell')(**spec)




def ebfit():
    pass

if __name__ == '__main__':
    singlefit()
