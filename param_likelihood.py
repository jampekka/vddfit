from fit_d2p import get_keio_trials, DT, vddm_params, mangle_single_tau, mangle_blocker_tau, model_params, actgrid
import scipy.stats
import numpy as np
import json
from crossmods import Vddm, Grid1d
from scipy.special import softmax
import matplotlib.pyplot as plt
import itertools
import pickle
from pprint import pprint
import copy
import sys

positive_params = 'std', 'scale', 'act_threshold'

def std_norm_logpdf(x):
    return np.log(1/np.sqrt(2*np.pi)) - 1/2*x**2

def std_norm_proposal():
    x = np.random.randn()
    l = std_norm_logpdf(x) 
    return x, l

def sample_keio_params():
    trials = get_keio_trials()
    modes = vddm_params['keio_uk']

    def likelihood(**params):
        lik = 0
        model = Vddm(dt=DT, **model_params(params))
        for trial in trials:
            if len(trial) == 3:
                traj, traj_b, rts = trial
                tau, tau_b = mangle_blocker_tau(traj, traj_b, **params)
                pdf = model.blocker_decisions(actgrid, tau, tau_b)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            else:
                traj, rts = trial
                tau = mangle_single_tau(traj, **params)
                pdf = model.decisions(actgrid, tau)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
        return lik
    
    def get_proposal():
        proposal_s = 2.0
        total = 0.0
        param = {}
        for n in 'std', 'damping', 'scale', 'act_threshold':
            x, l = std_norm_proposal()
            x *= proposal_s
            param[n] = np.exp(x + np.log(modes[n]))
            total += l
        for n in 'tau_threshold', 'pass_threshold', 'dot_coeff', 'dist_coeff':
            x, l = std_norm_proposal()
            x *= proposal_s
            param[n] = x + modes[n]
            total += l

        return param, total

    while True:
        prop, proplik = get_proposal()
        obs_lik = likelihood(**prop)
        lik = obs_lik - proplik
        print(json.dumps([lik, prop]))

initial_guess = dict(
    std=            1.0,
    damping=        0.0,
    scale=          1.0,
    tau_threshold=  2.0,
    act_threshold=  1.0,
    pass_threshold= 0.0,
    dot_coeff=      0.0,
    ehmi_coeff=     0.0,
    dist_coeff=     0.0,
)

def likelihooder(trials):
    def likelihood(**params):
        lik = 0
        model = Vddm(dt=DT, **model_params(params))
        for trial in trials:
            if len(trial) == 3:
                traj, traj_b, rts = trial
                #tau = mangle_tau_(traj, traj_b, **params)
                tau, tau_b = mangle_blocker_tau(traj, traj_b, **params)
                pdf = model.blocker_decisions(actgrid, tau, tau_b)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            else:
                traj, rts = trial
                tau = mangle_single_tau(traj, **params)
                pdf = model.decisions(actgrid, tau)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
        if np.isnan(lik):
            print("Nanlik", params)
            return -np.inf
        return lik
    return likelihood

def keio_mcmc():
    import pandas as pd
    from adametro import adaptive_metropolis
    trials = get_keio_trials()
    #initial = copy.copy(initial_guess)
    initial = copy.copy(vddm_params['keio_uk'])
    #del initial['ehmi_coeff']
    for k in positive_params:
        initial[k] = np.log(initial[k])
    keys, initial = zip(*initial.items())
    positive_idx = [i for i in range(len(keys)) if keys[i] in positive_params]
        
    likelihood = likelihooder(trials)
    def wrapper(x):
        unmangled = dict(zip(keys, x))
        for k in positive_params:
            unmangled[k] = np.exp(unmangled[k])
        return likelihood(**unmangled)
    
    sampler = adaptive_metropolis(wrapper, initial, cov=np.eye(len(initial))*0.001)
    samples = []
    metas = []
    for i, (sample, meta) in enumerate(sampler):
        sample = sample.copy()
        for pi in positive_idx:
            sample[pi] = np.exp(sample[pi])
        samples.append(sample)
        metas.append(meta)
        print(sample)
        pprint(meta)
        if i > 3000: break
    #samples, metas = zip(*itertools.islice(sampler, 100))
    samples = np.array(samples)
    samples = pd.DataFrame({k: v for k, v in zip(keys, samples.T)})
    samples.to_csv("keio_samples.csv", index=False)
    pickle.dump([metas, samples], open("keio_samples.pickle", 'wb'))

    plt.hist(samples['dist_coeff'])
    plt.show()

def differ(f, dx=np.sqrt(np.finfo(float).eps)):
    def wrapper(x):
        l = f(x)
        dl = np.empty_like(x)
        param = x.copy()
        for i in range(len(dl)):
            param[:] = x
            param[i] += dx
            l_ = f(param)
            dl[i] = (l_ - l)/dx
        return l, dl

    return wrapper

def keio_nuts():
    import pandas as pd
    import littlemcmc as lmc
    trials = get_keio_trials()
    #initial = copy.copy(initial_guess)
    initial = copy.copy(vddm_params['keio_uk'])
    del initial['ehmi_coeff']
    for k in positive_params:
        initial[k] = np.log(initial[k])
    keys, initial = zip(*initial.items())
    positive_idx = [i for i in range(len(keys)) if keys[i] in positive_params]
    
    likelihood = likelihooder(trials)
    def wrapper(x):
        unmangled = dict(zip(keys, x))
        for k in positive_params:
            unmangled[k] = np.exp(unmangled[k])
        return likelihood(**unmangled)
    l_and_ld = differ(wrapper)
    
    samples, stats = lmc.sample(l_and_ld, start=np.array(initial), cores=1, chains=1, tune=10, draws=10, model_ndim=len(keys))
    
    for dim in samples[0].T:
        plt.hist(dim)
        plt.show()
    return
    samples = []
    metas = []
    for i, (sample, meta) in enumerate(sampler):
        sample = sample.copy()
        for pi in positive_idx:
            sample[pi] = np.exp(sample[pi])
        samples.append(sample)
        metas.append(meta)
        print(sample)
        pprint(meta)
        if i > 3000: break
    #samples, metas = zip(*itertools.islice(sampler, 100))
    samples = np.array(samples)
    samples = pd.DataFrame({k: v for k, v in zip(keys, samples.T)})
    samples.to_csv("keio_samples.csv", index=False)
    pickle.dump([metas, samples], open("keio_samples.pickle", 'wb'))

    plt.hist(samples['dist_coeff'])
    plt.show()

colsyms = {
    'std': (r'$\sigma$', "Noise std"),
    'damping': (r'$\alpha$', "Damping"),
    'scale': (r'$m$', "Scale"),
    'tau_threshold': (r"$\tau'$", "Tau thres."),
    'act_threshold': (r"$A'$", "Evidence thres."),
    'pass_threshold': (r"$\tau_p$", "Passed thres."),
    'dot_coeff': (r"$\beta_{\dot\tau}$", r'$\dot\tau$ coeff.'),
    'dist_coeff': (r"$\beta_{D}$", "Distance coeff."),
        }

def plot_keio_marginals():
    import pandas as pd
    #weights, values = zip(*(json.loads(l) for l in open('keio_uk_samples.jsons')))
    #param = pd.DataFrame.from_records(values)
    #param['weights'] = weights
    #param = param[param['weights'].notna()]

    #param['weights'] = softmax(param['weights'].values)
    #param = param[param['weights'] > 0]
    #print(param['weights'].values)
    
    params = []
    for f in sys.argv[1:]:
        param = pd.read_csv(f)
        try:
            param = param.drop(columns="ehmi_coeff")
        except KeyError:
            pass
        param = param[1000:]
        params.append(param)
    param = pd.concat(params)

    modes = copy.copy(vddm_params['keio_uk'])
    #for col in param:
    #    plt.hist(param[col], bins=30)
    #    plt.title(col)
    #    plt.show()
    
    n = len(param.columns)
    rows = int(np.sqrt(n))
    cols = int(n/rows + 0.5)
    fig, axs = plt.subplots(rows, cols, constrained_layout=True)
    #axs = list(itertools.chain(axs))
    axs = axs.reshape(-1)
    for i, col in enumerate(param):
        ax = axs[i]
        ax.get_yaxis().set_visible(False)
        #ax.set_title(colsyms.get(col, col)[0])
        label = "%s %s"%colsyms.get(col, col)[::-1]
        ax.set_title(label)
        vals = param[col].values
        est = scipy.stats.gaussian_kde(vals)
        h, l = np.percentile(vals, [2.5, 97.5])
        m = modes[col]
        rng = np.linspace(np.min(vals), np.max(vals), 1000)
        ax.hist(vals, density=True, bins=20, color='black', alpha=0.25)
        ax.axvline(modes[col], color='black')
        ax.set_xticks([l, m, h])
        ax.set_xticklabels(np.round([l, m, h], 1))
        #plt.plot(rng, est(rng))
    plt.show()

def plot_keio_correlations():
    import pandas as pd
    from pandas.plotting import scatter_matrix
    #weights, values = zip(*(json.loads(l) for l in open('keio_uk_samples.jsons')))
    #param = pd.DataFrame.from_records(values)
    #param['weights'] = weights
    #param = param[param['weights'].notna()]

    #param['weights'] = softmax(param['weights'].values)
    #param = param[param['weights'] > 0]
    #print(param['weights'].values)
    
    params = []
    for f in sys.argv[1:]:
        param = pd.read_csv(f)
        try:
            param = param.drop(columns="ehmi_coeff")
        except KeyError:
            pass
        param = param[1000:]
        params.append(param)
    param = pd.concat(params)

    names = {k: v[0] for k, v in colsyms.items()}
    print(names)
    param = param.rename(columns=names)

    for name in param.columns:
        for oname in param.columns:
            if name == oname: continue
            print(scipy.stats.pearsonr(param[name], param[oname])[0])
    
    #modes = copy.copy(vddm_params['keio_uk'])
    scatter_matrix(param, color='black', hist_kwds={'color': 'black'})
    plt.show()





if __name__ == '__main__':
    #sample_keio_params()
    #plot_keio_marginals()
    plot_keio_correlations()
    #keio_mcmc()
    #keio_nuts()
