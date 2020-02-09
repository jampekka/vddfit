from itertools import groupby
import pandas as pd
import numpy as np
from vdd_disc import vdd_loss, vdd_decision_pdf, VddmParams, vdd_blocker_loss, vdd_blocker_decision_pdf
import tdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize
from pprint import pprint
import hikersim
from crossmods import Vddm, LognormalTdm as Tdm, Grid1d
from collections import defaultdict
from kwopt import minimizer, logbarrier, logitbarrier, fixed

#_minimizer = minimizer
#def minimizer(loss, **kwargs):
#    return _minimizer(loss, opt_f=scipy.optimize.basinhopping, minimizer_kwargs=kwargs)

actgrid = Grid1d(-3.0, 3.0, 100)
def fit_vdd(trials, dt):
    init = dict(
           std=1.0,
           damping=1.0,
           scale=1.0,
           tau_threshold=2.5,
           act_threshold=1.0,
           pass_threshold=0.0
           )


    spec = dict(
        std=            (init['std'], logbarrier),
        damping=        (init['damping'], logbarrier),
        scale=          (init['scale'], logbarrier,),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier,),
        pass_threshold= (0.0,)
            )
    
    def loss(**params):
        loss = 0.0
        model = Vddm(dt=dt, **params)
        for tau, cts in trials:
            mylik = model.decisions(actgrid, tau).loglikelihood(cts)
            loss -= mylik
        return loss
    
    def cb(x, f, accept):
        print(f, accept, x)
    return minimizer(loss,
            method='powell', #options={'maxiter': 1}
            )(**spec)
    
def fit_blocker_vdd(trials, dt):
    init = dict(
           std=1.0,
           damping=1.0,
           scale=1.0,
           tau_threshold=2.5,
           act_threshold=1.0,
           pass_threshold=0.0
           )

    spec = dict(
        std=            (init['std'], logbarrier),
        damping=        (init['damping'], logbarrier),
        scale=          (init['scale'], logbarrier),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier),
        pass_threshold= (1.0,)
            )
    
    def loss(**params):
        loss = 0.0
        model = Vddm(dt=dt, **params)
        #print(model.tau_threshold)
        for tau, tau_b, cts in trials:
            pdf = model.blocker_decisions(actgrid, tau, tau_b)
            myloss = pdf.loglikelihood(cts, slack=np.finfo(float).eps)
            #print(myloss)
            loss -= myloss
        return loss
    return minimizer(loss,
            method='powell', #options={'maxiter': 1}
            )(**spec)

def fit_tdm(trials, dt):
    spec = dict(
        thm=            (np.log(3.0),),
        ths=            (np.sqrt(1/6), logbarrier),
        lagm=            (np.log(0.3),),
        lags=            (np.sqrt(1/6), logbarrier),
        pass_th=          (1.0),
            )
    
    def loss(**kwargs):
        lik = 0
        model = Tdm(**kwargs)
        for taus, rts in trials:
            pdf = model.decisions(taus, dt)
            lik += pdf.loglikelihood(rts, np.finfo(float).eps)
        return -lik
    return minimizer(loss, method='powell', #options={'maxiter': 1}
            )(**spec)

def fit_blocker_tdm(trials, dt):
    spec = dict(
        thm=            (np.log(3.0),),
        ths=            (np.sqrt(1/6), logbarrier),
        lagm=            (np.log(0.3),),
        lags=            (np.sqrt(1/6), logbarrier),
        pass_th=          (1.0),
            )
    
    def loss(**kwargs):
        lik = 0
        model = Tdm(**kwargs)
        for taus, taus_b, rts in trials:
            pdf = model.blocker_decisions(taus, taus_b, dt)
            lik += pdf.loglikelihood(rts, np.finfo(float).eps)
        return -lik
    return minimizer(loss,
            method='powell', #options={'maxiter': 1}
            )(**spec)

def fit_hiker():
    data = pd.read_csv('hiker_cts.csv')
    data['has_hmi'] = data.braking_condition > 2

    data = data.query('braking_condition == 1')
    
    leader_start = 100
    DT = 1/30
    trials = []
    for g, d in data.groupby(['time_gap', 'speed', 'is_braking', 'has_hmi']):
        time_gap, speed, is_braking, has_hmi = g
        
        starttime = -leader_start/speed
        endtime = starttime + 20
        if not is_braking:
            endtime = time_gap

        ts = np.arange(starttime, endtime, DT)
        lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking)

        
        tau_lag = -lag_x/lag_speed
        lead_dist = leader_start - (ts - starttime)*speed
        tau_lead = lead_dist/speed
        
        crossing_times = d.crossing_time.values - starttime
        crossing_times[~np.isfinite(crossing_times)] = np.inf
        trials.append((tau_lag, tau_lead, crossing_times))

    vdd_fit = fit_blocker_vdd(trials, DT)
    vdd_params = vdd_fit.kwargs
    print("VDD")
    print(vdd_fit)
    tdm_fit = fit_blocker_tdm(trials, DT)
    tdm_params = tdm_fit.kwargs
    print("TDM")
    print(tdm_fit)
    #print(fit)

    #param = {'std': 0.48014622086528025, 'damping': 1.431032217574059, 'scale': 1.0, 'tau_threshold': 2.1554361707966425, 'act_threshold': 1.0, 'pass_threshold': 0.9657682789027324}
    
    output = PdfPages("hikerfit.pdf")
    def show():
        output.savefig()
        plt.close()
    
    for tau, btau, cts in trials:
        ts = np.arange(0, len(tau))*DT
        #param['dt'] = DT
        #pdf = vdd_blocker_decision_pdf(VddmParams(**param), tau, btau)
        vdd_pdf = Vddm(dt=DT, **vdd_params).blocker_decisions(actgrid, tau, btau)
        tdm_pdf = Tdm(**tdm_params).blocker_decisions(tau, btau, DT)
        plt.plot(ts, tau)
        plt.twinx()
        plt.plot(ts, np.vectorize(vdd_pdf)(ts + DT/2)/(1 - vdd_pdf.uncrossed), label=f"VDDM, noncrossing {100*vdd_pdf.uncrossed:.0f}%")
        noncrossing = np.sum(~np.isfinite(cts))/len(cts)
        plt.plot(ts, np.vectorize(tdm_pdf)(ts + DT/2)/(1 - tdm_pdf.uncrossed), label=f"TDM, noncrossing {100*tdm_pdf.uncrossed:.0f}%")
        noncrossing = np.sum(~np.isfinite(cts))/len(cts)

        plt.hist(cts[np.isfinite(cts)] + DT/2, bins=np.arange(0, ts[-1], 0.2), density=True,
            label=f"Empirical, noncrossing {100*noncrossing:.0f}%")
        plt.axvline(scipy.interpolate.interp1d(btau, ts)(0), color='black', label='Leader passed')
        plt.legend()

        show()
    
    output.close()


def fit_keio(country='uk'):
    all_trajectories = pd.read_csv('d2p_trajectories.csv')
    all_responses = pd.read_csv(f'd2p_cross_times_{country}.csv')
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
        trials[trial] = (tau, resp)
        trial_taus[trial] = tau

    for s, sd in all_responses.groupby('subject_id'):
        st_rts = subj_trial_responses[s] = {}
        for t, td in sd.groupby('trial_id'):
            if t not in trial_taus: continue
            st_rts[t] = td.cross_time.values

    #dt = np.median(np.diff(all_trajectories['time_c']))
    dt = 1/30

    #param = {'std': 0.6970908298337709, 'damping': 0.46041074483528815, 'scale': 1.0, 'tau_threshold': 2.0542953634220624, 'act_threshold': 1.0}
#param = {'std': 0.6944196353260089, 'damping': 0.4747255460581086, 'scale': 0.7848596401338991, 'tau_threshold': 1.984151857035362, 'act_threshold': 1.0}

    loss = vdd_loss(list(trials.values()), dt, N=100)
    liks = []


    vdd_fit = fit_vdd(list(trials.values()), dt)
    print("VDD")
    print(vdd_fit)
    tdm_fit = fit_tdm(list(trials.values()), dt)
    print("TDM")
    print(tdm_fit)
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

    pdf = PdfPages(f"keiofit_{country}.pdf")
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
        kwargs = vdd_fit.kwargs
        kwargs['dt'] = dt
        #vdd = vdd_decision_pdf(VddmParams(**kwargs), tau)
        vdd_pdf = Vddm(**kwargs).decisions(actgrid, tau)
        plt.plot(ts, np.vectorize(vdd_pdf)(ts + dt/2), label='VDDM')
        
        #tdm_pdf = tdm.predict_decision_times(dt, tau, **tdmfit.kwargs)(ts)
        tdm_pdf = Tdm(**tdm_fit.kwargs).decisions(tau, dt)
        plt.plot(ts, np.vectorize(tdm_pdf)(ts + dt/2), label='TDM')
        plt.ylabel("Crossing likelihood")
        plt.legend()
        plt.twinx()
        plt.plot(ts, tau, color='black', label='Tau')
        plt.ylabel("Tau")
        plt.xlabel("Time")
        plt.ylim(0, 10)
        show()

    pdf.close()

def ecdf(vs):
    vs = np.sort(vs)
    x = [-np.inf]
    cdf = [0]
    for v in vs:
        if not np.isfinite(v): continue
        if v == x[-1]:
            cdf[-1] += 1
        if v > x[-1]:
            cdf.append(cdf[-1] + 1)
            x.append(v)

    x = np.array(x)
    cdf = np.array(cdf, dtype=float)
    cdf /= len(vs)
    return scipy.interpolate.interp1d(x, cdf, kind='previous', fill_value='extrapolate')

vddm_params = {
    'keio_uk': {'std': 0.6895813463269477, 'damping': 1.465877152701145, 'scale': 0.6880628317112305, 'tau_threshold': 2.267405295940994, 'act_threshold': 0.976580507453147, 'pass_threshold': -0.0077883119117825046},
    'keio_japan': {'std': 0.6200239180599325, 'damping': 1.7363852589095383, 'scale': 0.34685222829243945, 'tau_threshold': 2.179287004380974, 'act_threshold': 0.9509308457583107, 'pass_threshold': -0.016836928176652676},
    'hiker': {'std': 0.5229662639983371, 'damping': 0.9946376439854723, 'scale': 0.48663037841127904, 'tau_threshold': 2.308618665956897, 'act_threshold': 1.053645908407938, 'pass_threshold': 1.024761904761648}

    }
tdm_params = {
    'keio_uk': {'thm': 1.4822542031746122, 'ths': 0.4160580398784046, 'lagm': -0.018037627273216346, 'lags': 0.5394395959794362, 'pass_th': -0.07255736056398886},
    'keio_japan': {'thm': 1.7737959051473795, 'ths': 0.4270654760140321, 'lagm': 0.15978825057281343, 'lags': 0.5678380232026445, 'pass_th': -0.015453144629448029},
    'hiker': {'thm': 1.58413762002706, 'ths': 0.30787584232846543, 'lagm': 0.18325303499450624, 'lags': 0.24160198818000903, 'pass_th': 0.9960908329751528}

}

def plot_keio(country='uk'):
    dt = 1/30
    all_trajectories = pd.read_csv('d2p_trajectories.csv')
    all_responses = pd.read_csv(f'd2p_cross_times_{country}.csv')
    all_responses[["subject_id", "trial_number"]] = all_responses.unique_ID.str.split("_", 1, expand=True)
    responses = dict(list(all_responses.groupby('trial_id')))
    
    vddm = Vddm(dt=dt, **vddm_params[f'keio_{country}'])
    tdm = Tdm(**tdm_params[f'keio_{country}'])
    
    pdf = PdfPages(f"keiofit_{country}.pdf")
    def show():
        pdf.savefig()
        plt.close()

    tau_types = defaultdict(list)
    for trial, td in all_trajectories.groupby('trial_n'):
        if trial == 1: continue
        has_decel = np.std(td.speed.values) > 0.01
        tau_type = trial
        if not has_decel:
            tau_type = round(td.distance.values[0]/td.speed.values[0], 1)
        
        print(tau_type, has_decel)
        tau_types[(tau_type, has_decel)].append((trial, td))
    
    stats = []
    #for trial, td in all_trajectories.groupby('trial_n'):
    for (tau_type, has_decel), trials in tau_types.items():
        tids, _ = zip(*trials)
        for trial, td in trials:
            #plt.plot(td.time_c, td.distance)
            ts = td.time_c.values
            tr = responses[trial]
            tau = td.distance/td.speed
            vddm_pdf = vddm.decisions(actgrid, tau.values)
            tdm_pdf = tdm.decisions(tau.values, dt)

            stats.append(dict(
                mean=np.mean(tr.cross_time.values),
                mean_vdd=np.dot(np.array(vddm_pdf.ps)*dt/(1 - vddm_pdf.uncrossed), ts),
                mean_tdm=np.dot(np.array(tdm_pdf.ps)*dt/(1 - tdm_pdf.uncrossed), ts),
                has_accel=np.std(td.speed) > 0.01
                ))

            label = "Empirical v0 {td.speed[0]:.1f} m/s"
            plt.plot(td.time_c.values, ecdf(tr.cross_time)(td.time_c.values)*100, label=f'Empirical, d0={td.distance.values[0]:.1f} m')
        plt.title(f"Keio {country} trial type {' and '.join(map(str, tids))}")
        plt.plot(td.time_c.values, np.cumsum(np.array(vddm_pdf.ps)*dt)*100, 'k', label='VDDM')
        plt.plot(td.time_c.values, np.cumsum(np.array(tdm_pdf.ps)*dt)*100, 'k--', label='TDM')
        plt.legend()
        plt.xlim(ts[0], ts[-1])
        plt.ylim(-1, 101)
        plt.ylabel('Percentage crossed')
        plt.xlabel('Time (seconds)')
        plt.twinx()
        plt.plot(ts, tau, label='Time to arrival', color='black', alpha=0.5)
        plt.ylabel('Time to arrival (seconds)')
        plt.ylim(0, 8)
        show()

    stats = pd.DataFrame.from_records(stats)
    vstats = stats[~stats.has_accel]
    plt.plot(vstats['mean'], vstats.mean_vdd, 'C0o', label='VDDM (constant speed)')
    plt.plot(vstats['mean'], vstats.mean_tdm, 'C1o', label='TDM (constant speed)')
    
    vstats = stats[stats.has_accel]
    plt.plot(vstats['mean'], vstats.mean_vdd, 'C0x', label='VDDM (variable speed)')
    plt.plot(vstats['mean'], vstats.mean_tdm, 'C1x', label='TDM (variable speed)')
    
    rng = stats['mean'].min(), stats['mean'].max()
    plt.plot(rng, rng, 'k-', alpha=0.3)
    plt.legend()
    plt.xlabel('Measured mean crossing time (seconds)')
    plt.ylabel('Predicted mean crossing time (seconds)')
    plt.axis('equal')
    show()
    pdf.close()

        
def plot_hiker():
    dt = 1/30
   
    data = pd.read_csv('hiker_cts.csv')
    data['has_hmi'] = data.braking_condition > 2
    data = data.query('braking_condition <= 3')
    
    pdf = PdfPages(f"hikerfit.pdf")
    def show():
        pdf.savefig()
        plt.close()
    
    leader_start = 100
    DT = 1/30
    trials = []
    trajectories = []
    for i, (g, d) in enumerate(data.groupby(['time_gap', 'speed', 'is_braking', 'has_hmi'])):
        time_gap, speed, is_braking, has_hmi = g
        
        starttime = -leader_start/speed
        endtime = starttime + 20
        if not is_braking:
            endtime = time_gap

        #ts = np.arange(starttime, endtime, DT)
        ts = np.arange(-5, min(8, endtime), DT)
        lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking)

        tau_lag = -lag_x/lag_speed
        lead_dist = leader_start - (ts - starttime)*speed
        tau_lead = lead_dist/speed
        
        crossing_times = d.crossing_time.values #- starttime
        crossing_times[~np.isfinite(crossing_times)] = np.inf
        trajectories.append(dict(
            trial_id=i,
            ts=ts, freetime=starttime, speed=lag_speed,
            tau_lag=tau_lag, tau_lead=tau_lead, crossing_times=crossing_times,
            lag_distance=lag_x,
            t_brake=t_brake, t_stop=t_stop,
            time_gap=time_gap, is_braking=is_braking, has_hmi=has_hmi))
    
    vddm = Vddm(dt=dt, **vddm_params[f'hiker'])
    tdm = Tdm(**tdm_params[f'hiker'])
    stats = []
    # TODO: This doesn't work!
    
    def key(x):
        if x['is_braking']:
            return (1, x['trial_id'])
        return (0, round(x['time_gap'], 1))

    for _, sametau in groupby(sorted(trajectories, key=key), key=key):
        for i, trial in enumerate(sametau):
            #plt.plot(td.time_c, td.distance)
            plt.title(_)
            #plt.title(f"HIKER v0 {trial['speed'][0]:.1f} m/s")
            ts = trial['ts']
            tr = trial['crossing_times']
            tau = trial['tau_lag']
            tau_b = trial['tau_lead']
            vddm_pdf = vddm.blocker_decisions(actgrid, tau, tau_b)
            tdm_pdf = tdm.blocker_decisions(tau, tau_b, dt)

            cdf = ecdf(tr)(ts)
            cdf_vdd = np.cumsum(np.array(vddm_pdf.ps)*dt)
            cdf_tdm = np.cumsum(np.array(tdm_pdf.ps)*dt)


            early_t = np.min(ts[(tau_b > 0) & (tau <= 0)], initial=np.inf)
            early_t = min(trial['t_stop'], early_t, ts[-1])
            early_i = np.searchsorted(ts, early_t)
            stats.append(dict(
                mean=np.mean(tr[np.isfinite(tr)]),
                mean_vdd=np.dot(np.array(vddm_pdf.ps)*dt/(1 - vddm_pdf.uncrossed), ts),
                mean_tdm=np.dot(np.array(tdm_pdf.ps)*dt/(1 - tdm_pdf.uncrossed), ts),
                has_accel=np.std(trial['speed']) > 0.01
                ))
            
            d0 = -scipy.interpolate.interp1d(ts, trial['lag_distance'])(0)
            plt.plot(ts, ecdf(tr)(ts)*100, label=f'Empirical, d0={d0:.1f} m')
        
        plt.plot(ts, np.cumsum(np.array(vddm_pdf.ps)*dt)*100, 'k', label='VDDM')
        plt.plot(ts, np.cumsum(np.array(tdm_pdf.ps)*dt)*100, 'k--', label='TDM')


        plt.xlim(-1, ts[-1])
        plt.ylim(-1, 101)
        plt.ylabel('Percentage crossed')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.twinx()
        plt.ylabel('Time to arrival (seconds)')
        plt.plot(ts, tau, label='Time to arrival', color='black', alpha=0.5)
        plt.plot(ts, tau_b, 'k--', alpha=0.5)
        plt.ylim(0, 8)
        show()

    stats = pd.DataFrame.from_records(stats)
    vstats = stats[~stats.has_accel]
    plt.plot(vstats['mean'], vstats.mean_vdd, 'C0o', label='VDDM (constant speed)')
    plt.plot(vstats['mean'], vstats.mean_tdm, 'C1o', label='TDM (constant speed)')
    
    vstats = stats[stats.has_accel]
    plt.plot(vstats['mean'], vstats.mean_vdd, 'C0x', label='VDDM (variable speed)')
    plt.plot(vstats['mean'], vstats.mean_tdm, 'C1x', label='TDM (variable speed)')
    
    rng = stats['mean'].min(), stats['mean'].max()
    plt.plot(rng, rng, 'k-', alpha=0.3)
    plt.legend()
    plt.xlabel('Measured mean crossing time (seconds)')
    plt.ylabel('Predicted mean crossing time (seconds)')
    plt.axis('equal')
    show()

    pdf.close()

if __name__ == '__main__':
    plot_hiker()
    #plot_keio('japan')
    #plot_keio('uk')
    #print("keiouk")
    #fit_keio('uk')
    #print("keiojapan")
    #fit_keio('japan')
    #print("hiker")
    #fit_hiker()
