import pandas as pd
import numpy as np
from vdd import vdd_loss, simulate_times, sample_lik
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
       std=1.0,
       damping=2.5,
       scale=1.0,
       tau_threshold=3.5,
       act_threshold=1.0
       )

loss = vdd_loss(list(trials.values()), dt)
liks = []

griddens = 100
stds = np.linspace(3, 8, griddens)
#for std in stds:
#    liks.append(-loss(**{**param, **{'std': std}}))
#plt.plot(stds/np.sqrt(dt), liks)

thresholds = np.linspace(2.0, 5.5, griddens)

S, T = np.meshgrid(stds, thresholds)
for std, threshold in zip(*(x.flat for x in (S, T))):
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
    noise = np.random.randn(10000, len(tau))
    sample = simulate_times(tau, dt, noise, **param)
    ts = np.arange(len(tau))*dt
    plt.title(f"Trial type {trial}")
    plt.hist(resp, bins=20, density=True, label='Measurements')
    #plt.hist(sample[np.isfinite(sample)], bins=100, histtype='step', density=True)
    plt.plot(ts, sample_lik(ts, sample, dt), label='Model')
    plt.ylabel("Crossing likelihood")
    plt.twinx()
    plt.plot(ts, tau, color='black', label='Tau')
    plt.ylabel("Tau")
    plt.xlabel("Time")
    plt.ylim(0, 10)
    plt.legend()
    show()

pdf.close()
