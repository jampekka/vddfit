import pandas as pd
import numpy as np
from vdd_disc import vdd_loss, vdd_pdf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

all_trajectories = pd.read_csv('d2p_trajectories.csv')
all_responses = pd.read_csv('d2p_cross_times_uk.csv')


responses = dict(list(all_responses.groupby('trial_id')))

trials = {}

dt = np.median(np.diff(all_trajectories['time_c']))

for trial, traj in all_trajectories.groupby('trial_n'):
    if np.std(traj.speed) > 0.01:
        continue
    resp = responses[trial]['cross_time'].values
    #tau = traj['tau'].values
    tau = traj['distance'].values/traj['speed'].values
    tau[tau < 0] = 1e50
    trials[trial] = (tau, resp)

param = dict(
       std=0.9,
       damping=0.4,
       scale=1.0,
       tau_threshold=4.3,
       act_threshold=1.0
       )

loss = vdd_loss(list(trials.values()), dt)
liks = []

from kwopt import minimizer, logbarrier, logitbarrier, fixed
def fit_vdd(trials, dt, init=None):
    if init is None:
        init = param.copy()

    spec = dict(
        std=            (init['std'], logbarrier),
        damping=      (init['damping'], logitbarrier, fixed),
        scale=          (init['scale'], logbarrier, fixed),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier, fixed)
            )
    
    loss = vdd_loss(trials, dt)
    return minimizer(loss, method='nelder-mead')(**spec)

#print(fit_vdd(list(trials.values()), dt))
#asdfasf

griddens = 100
stds = np.linspace(0.8, 1.2, griddens)
thresholds = np.linspace(3.0, 5.0, griddens)
#for std in stds:
#    liks.append(-loss(**{**param, **{'std': std}}))
#plt.plot(stds, liks)
#for th in thresholds:
#    liks.append(-loss(**{**param, **{'tau_threshold': th}}))
#plt.plot(thresholds, liks)

S, T = np.meshgrid(stds, thresholds)
N = np.product(S.shape)
for i, (std, threshold) in enumerate(zip(*(x.flat for x in (S, T)))):
    print(f"{i/N*100}% done")
    liks.append(-loss(**{**param, **{'tau_threshold': threshold, 'std': std}}))

liks = np.array(liks)
winner = np.nanargmax(liks)
std, threshold = S.flat[winner], T.flat[winner]
param = {**param, **{'tau_threshold': threshold, 'std': std}}


pdf = PdfPages("gridfit.pdf")
def show():
    pdf.savefig()
    plt.close()

plt.title("Parameterization likelihood")
plt.pcolormesh(S, T, np.exp(liks.reshape(S.shape)))
plt.plot(std, threshold, 'ro', label='Maximum likelihood')
plt.xlabel("Noise std")
plt.ylabel("Tau threshold")
plt.colorbar()
plt.legend()
show()

for trial, (tau, resp) in trials.items():
    ts = np.arange(len(tau))*dt
    plt.title(f"Trial type {trial}")
    plt.hist(resp, bins=20, density=True, label='Measurements')
    #plt.hist(sample[np.isfinite(sample)], bins=100, histtype='step', density=True)
    cross_pdf = vdd_pdf(tau, dt, **param)
    plt.plot(ts, cross_pdf, label='Model')
    plt.ylabel("Crossing likelihood")
    plt.twinx()
    plt.plot(ts, tau, color='black', label='Tau')
    plt.ylabel("Tau")
    plt.xlabel("Time")
    plt.ylim(0, 10)
    plt.legend()
    show()

pdf.close()
