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



vddm_params = {
    'keio_uk': {'std': 0.6895813463269477, 'damping': 1.465877152701145, 'scale': 0.6880628317112305, 'tau_threshold': 2.267405295940994, 'act_threshold': 0.976580507453147, 'pass_threshold': -0.0077883119117825046},
    'keio_japan': {'std': 0.6200239180599325, 'damping': 1.7363852589095383, 'scale': 0.34685222829243945, 'tau_threshold': 2.179287004380974, 'act_threshold': 0.9509308457583107, 'pass_threshold': -0.016836928176652676},
    #'hiker': {'std': 0.5229662639983371, 'damping': 0.9946376439854723, 'scale': 0.48663037841127904, 'tau_threshold': 2.308618665956897, 'act_threshold': 1.053645908407938, 'pass_threshold': 1.024761904761648},
    #'hiker': {'std': 0.5935959086407167, 'damping': 0.9151503732988887, 'scale': 0.39910022137223705, 'tau_threshold': 2.533825817019247, 'act_threshold': 1.0588392695282198, 'pass_threshold': 0.9900001990325101, 'dot_coeff': 1.1290275837238286, 'ehmi_coeff': 0.33449603099173664},
    #'hiker': {'std': 0.6242299905405555, 'damping': 0.8077397267051836, 'scale': 0.37777720233236123, 'tau_threshold': 3.0037091915789738, 'act_threshold': 1.0, 'pass_threshold': 0.9568042787179346, 'dot_coeff': 1.517345748330563, 'ehmi_coeff': 0.4127020530271129, 'dist_coeff': 0.29191574142779203}
    #'hiker': {'std': 0.6737559893535666, 'damping': 0.7602189090471121, 'scale': 0.4657452690259603, 'tau_threshold': 3.158682267433862, 'act_threshold': 1.1100000438010933, 'pass_threshold': 0.9871867080244127, 'dot_coeff': 1.8687594289211251, 'ehmi_coeff': 0.36066282711053776, 'dist_coeff': 0.3196793196332367},
    #'hiker': {'std': np.exp(-0.43926672), 'damping': np.exp(-2.33382131), 'scale': np.exp(-1.27401909), 'tau_threshold': np.exp(1.71610424), 'act_threshold': np.exp(0.35830021), 'pass_threshold': 1.67203308, 'dot_coeff': 5.60317922, 'ehmi_coeff': 0.4471767, 'dist_coeff': 0.40036073},
    #'hiker': {'std': 0.6280571068112765, 'damping': 0.8603387229027513, 'scale': 0.44441779584286256, 'tau_threshold': 2.732806932418672, 'act_threshold': 1.1120349553671918, 'pass_threshold': 1.376707085329192, 'dot_coeff': 1.7494757571462418, 'ehmi_coeff': 0.34796227462413976, 'dist_coeff': 0.4158610927992449},

    #'hiker': {'std': 0.6426495118602893, 'damping': 0.09796207737160328, 'scale': 0.27855577718398583, 'tau_threshold': 5.556581951608067, 'act_threshold': 1.4279091034209075, 'pass_threshold': 1.672125243901112, 'dot_coeff': 5.584069278738003, 'ehmi_coeff': 0.45141101837739356, 'dist_coeff': 0.4008133595156099}

    'hiker': {'std': 0.6384974134032458, 'damping': 0.10186865847575911, 'scale': 0.2791830034375026, 'tau_threshold': 5.544952926633872, 'act_threshold': 1.4194944054263028, 'pass_threshold': 1.6722899150569444, 'dot_coeff': 5.5075638112727745, 'ehmi_coeff': 0.8482795240546034, 'dist_coeff': 0.4016305596474629}
    }
tdm_params = {
    'keio_uk': {'thm': 1.4822542031746122, 'ths': 0.4160580398784046, 'lagm': -0.018037627273216346, 'lags': 0.5394395959794362, 'pass_th': -0.07255736056398886},
    'keio_japan': {'thm': 1.7737959051473795, 'ths': 0.4270654760140321, 'lagm': 0.15978825057281343, 'lags': 0.5678380232026445, 'pass_th': -0.015453144629448029},
    #'hiker': {'thm': 1.58413762002706, 'ths': 0.30787584232846543, 'lagm': 0.18325303499450624, 'lags': 0.24160198818000903, 'pass_th': 0.9960908329751528},
    #'hiker': {'thm': 1.5152455089318642, 'ths': 0.32602324226593116, 'lagm': 0.1245117672647987, 'lags': 0.6021881227736422, 'pass_th': 0.7477010070264796, 'dot_coeff': 2.6937678922001163, 'ehmi_coeff': 0.2805016925833088}
    #'hiker': {'thm': 1.5442566242657068, 'ths': 0.3166462662796281, 'lagm': 0.20509904820328212, 'lags': 0.5482381938105031, 'pass_th': 0.8428738656551962, 'dot_coeff': 3.207119275055747, 'ehmi_coeff': 0.36038229654487886, 'dist_coeff': 0.2233817144842581}

    'hiker': {'thm': 1.5589096041551214, 'ths': 0.31723512786804214, 'lagm': 0.22346854221177187, 'lags': 0.5274406179540562, 'pass_th': 0.8792264031648431, 'dot_coeff': 3.2118358447257203, 'ehmi_coeff': 0.91032775986075, 'dist_coeff': 0.19840058606111935}
}


def mangle_tau(traj, dist_coeff=0.0, dot_coeff=0.0, ehmi_coeff=0.0, **kwargs):
    tau = traj['tau']
    prior_tau = traj['distance']/(50/3.6)
    return dist_coeff*(prior_tau - tau) + tau + dot_coeff*(traj['tau_dot'] + 1) + ehmi_coeff*traj['ehmi']

def model_params(params):
    return {k: v for k, v in params.items() if k not in ('dist_coeff', 'dot_coeff', 'ehmi_coeff')}

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
        pass_threshold= (0.0,),
        dot_coeff=      (0.001, logbarrier)
            )
    
    def loss(**params):
        loss = 0.0
        model = Vddm(dt=dt, **model_params(params))
        for traj, cts in trials:
            tau = mangle_tau(traj, **params)
            mylik = model.decisions(actgrid, tau).loglikelihood(cts)
            loss -= mylik
        return loss
    
    def cb(x, f, accept):
        print(f, accept, x)
    return minimizer(loss,
            method='powell', #options={'maxiter': 1}
            )(**spec)
    
def fit_blocker_vdd(trials, dt, init=vddm_params['hiker']):
    spec = dict(
        std=            (init['std'], logbarrier),
        damping=        (init['damping'], logbarrier),
        scale=          (init['scale'], logbarrier),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier),# (init['act_threshold'], logbarrier),
        pass_threshold= (init['pass_threshold']),
        dot_coeff=      (init['dot_coeff'],),
        ehmi_coeff=      (init['ehmi_coeff'],),
        dist_coeff=     (init['dist_coeff'],)
            )
    
    def loss(**params):
        loss = 0.0
        model = Vddm(dt=dt, **model_params(params))
        #print(model.tau_threshold)
        for traj, traj_b, cts in trials:
            tau = mangle_tau(traj, **params)
            tau_b = mangle_tau(traj_b, **params)
            pdf = model.blocker_decisions(actgrid, tau, tau_b)
            myloss = pdf.loglikelihood(cts - traj.time[0], slack=np.finfo(float).eps)
            loss -= myloss
        print(loss)
        return loss
    def cb(x, f, accept):
        print(f, accept, x)
    
    return minimizer(loss, method='powell')(**spec)
    #return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
    #        callback=cb, minimizer_kwargs={'method': 'powell'}
    #        #method='powell', #options={'maxiter': 1}
    #        )(**spec)

def fit_tdm(trials, dt):
    spec = dict(
        thm=            (np.log(3.0),),
        ths=            (np.sqrt(1/6), logbarrier),
        lagm=            (np.log(0.3),),
        lags=            (np.sqrt(1/6), logbarrier),
        pass_th=          (1.0),
        dot_coeff=      (0.001, logbarrier)
            )
    
    def loss(**params):
        lik = 0
        model = Tdm(**model_params(params))
        for traj, rts in trials:
            tau = mangle_tau(traj, **params)
            pdf = model.decisions(tau, dt)
            lik += pdf.loglikelihood(rts, np.finfo(float).eps)
        return -lik
    return minimizer(loss, method='powell', #options={'maxiter': 1}
            )(**spec)

def fit_blocker_tdm(trials, dt, init=tdm_params['hiker']):
    spec = dict(
        thm=            (init['thm'],),
        ths=            (init['ths'], logbarrier),
        lagm=            (init['lagm'],),
        lags=            (init['lags'], logbarrier),
        pass_th=          (init['pass_th']),
        dot_coeff=      (init['dot_coeff'],),
        ehmi_coeff=      (init['ehmi_coeff'],),
        dist_coeff=     (0.0,),
            )
    
    def loss(**params):
        lik = 0
        model = Tdm(**model_params(params))
        for traj, traj_b, rts in trials:
            tau = mangle_tau(traj, **params)
            tau_b = mangle_tau(traj_b, **params)
            pdf = model.blocker_decisions(tau, tau_b, dt)
            lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
        return -lik
    
    def cb(x, f, accept):
        print(f, accept, x)
    
    #return minimizer(loss, method='powell')(**spec)
    return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)

def fit_hiker():
    data = pd.read_csv('hiker_cts.csv')

    #data = data.query('braking_condition == 1')
    data = data.query('braking_condition <= 3')
    
    leader_start = 100
    DT = 1/30
    trials = []
    for g, d in data.groupby(['time_gap', 'speed', 'is_braking', 'has_ehmi']):
        time_gap, speed, is_braking, has_hmi = g
        
        """
        starttime = -leader_start/speed
        endtime = starttime + 20
        if not is_braking:
            endtime = time_gap

        ts = np.arange(starttime, endtime, DT)
        lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking)

        
        tau_lag = -lag_x/lag_speed
        tau_lag[~np.isfinite(tau_lag)] = 1e6
        lead_dist = leader_start - (ts - starttime)*speed
        tau_lead = lead_dist/speed
        tau_lead[~np.isfinite(tau_lead)] = 1e6
        
        crossing_times = d.crossing_time.values - starttime
        crossing_times[~np.isfinite(crossing_times)] = np.inf
        
        lead_traj = np.rec.fromarrays(
                (lead_dist,tau_lead, np.gradient(tau_lead, DT), np.zeros(len(ts))),
                names="distance,tau,tau_dot,ehmi")
        
        ehmi = np.zeros(len(ts))
        if has_hmi:
            ehmi[ts >= t_brake] = 1.0
        """
        
        crossing_times = d.crossing_time.values
        crossing_times[~np.isfinite(crossing_times)] = np.inf

        traj, lead_traj = get_trajectory(time_gap, speed, is_braking, has_hmi)
        
        #traj = np.rec.fromarrays((-lag_x,tau_lag, np.gradient(tau_lag, DT), ehmi), names="distance,tau,tau_dot,ehmi")
        trials.append((traj, lead_traj, crossing_times))

    #vdd_fit = fit_blocker_vdd(trials, DT)
    #vdd_params = vdd_fit.kwargs
    #print("VDD")
    #print(vdd_fit)
    
    tdm_fit = fit_blocker_tdm(trials, DT)
    tdm_params = tdm_fit.kwargs
    print("TDM")
    print(tdm_fit)


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
        traj['ehmi'] = False
        tau = traj['distance'].values/traj['speed'].values
        trials[trial] = (traj.to_records(), resp)

    for s, sd in all_responses.groupby('subject_id'):
        st_rts = subj_trial_responses[s] = {}
        for t, td in sd.groupby('trial_id'):
            if t not in trials: continue
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

def plot_keio(country='uk'):
    dt = 1/30
    all_trajectories = pd.read_csv('d2p_trajectories.csv')
    all_responses = pd.read_csv(f'd2p_cross_times_{country}.csv')
    all_responses[["subject_id", "trial_number"]] = all_responses.unique_ID.str.split("_", 1, expand=True)
    responses = dict(list(all_responses.groupby('trial_id')))
    
    #vddp = vddm_params[f'keio_{country}']
    #tdmp = tdm_params[f'keio_{country}']
    
    vddp = vddm_params['hiker']
    tdmp = tdm_params['hiker']

    vddm = Vddm(dt=dt, **model_params(vddm_params['hiker']))
    tdm = Tdm(**model_params(tdm_params['hiker']))

    #vddm = Vddm(dt=dt, **vddm_params[f'keio_{country}'])
    #tdm = Tdm(**tdm_params[f'keio_{country}'])
    
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
            td['ehmi'] = False
            tau_vddm = mangle_tau(td.to_records(), **vddp)
            vddm_pdf = vddm.decisions(actgrid, tau_vddm)
            tau_tdm = mangle_tau(td.to_records(), **tdmp)
            tdm_pdf = tdm.decisions(tau_tdm, dt)

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
    data = data.query('braking_condition <= 3')
    
    pdf = PdfPages(f"hikerfit.pdf")
    def show():
        pdf.savefig()
        plt.close()
    
    leader_start = 100
    DT = 1/30
    trials = []
    trajectories = []
    for i, (g, d) in enumerate(data.groupby(['time_gap', 'speed', 'is_braking', 'has_ehmi'])):
        time_gap, speed, is_braking, has_hmi = g
        
        starttime = -leader_start/speed
        endtime = starttime + 20
        if not is_braking:
            endtime = time_gap

        crossing_times = d.crossing_time.values #- starttime
        crossing_times[~np.isfinite(crossing_times)] = np.inf
        #ts = np.arange(starttime, endtime, DT)
        
        """
        ts = np.arange(-5, min(8, endtime), DT)
        lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking)

        tau_lag = -lag_x/lag_speed
        tau_lag[~np.isfinite(tau_lag)] = np.inf

        lead_dist = leader_start - (ts - starttime)*speed
        tau_lead = lead_dist/speed
        tau_lead[~np.isfinite(tau_lead)] = np.inf
        
        crossing_times = d.crossing_time.values - starttime
        crossing_times[~np.isfinite(crossing_times)] = np.inf
        
        lead_traj = np.rec.fromarrays(
                (lead_dist, tau_lead, np.gradient(tau_lead, DT), np.zeros(len(ts))),
                names="distance,tau,tau_dot,ehmi")
        
        ehmi = np.zeros(len(ts))
        if has_hmi:
            ehmi[ts >= t_brake] = 1.0
        
        traj = np.rec.fromarrays((-lag_x,tau_lag, np.gradient(tau_lag, DT), ehmi), names="distance,tau,tau_dot,ehmi")

        """

        traj, lead_traj = get_trajectory(time_gap, speed, is_braking, has_hmi)
        trajectories.append(dict(
            traj=traj, traj_lead=lead_traj,
            trial_id=i,
            #ts=ts, freetime=starttime, speed=lag_speed,
            initial_speed=speed,
            #tau_lag=tau_lag, tau_lead=tau_lead,
            crossing_times=crossing_times,
            #lag_distance=lag_x,
            #t_brake=t_brake, t_stop=t_stop,
            time_gap=time_gap, is_braking=is_braking, has_hmi=has_hmi))
    
    vddm = Vddm(dt=dt, **model_params(vddm_params[f'hiker']))
    tdm = Tdm(**model_params(tdm_params[f'hiker']))
    stats = []
    
    def key(x):
        time_gap = round(x['time_gap'], 1)
        initial_speed = round(x['initial_speed'], 1)
        #if not x['is_braking']:
        #    initial_speed = -1 # Hack!
        return (x['is_braking'], initial_speed, time_gap, x['has_hmi'])

    allpreds = []
    allpreds_tdm = []
    allcts = []
    for _, sametau in groupby(sorted(trajectories, key=key), key=key):
        for i, trial in enumerate(sametau):
            #plt.plot(td.time_c, td.distance)
            plt.title(_)
            #plt.title(f"HIKER v0 {trial['speed'][0]:.1f} m/s")
            ts = trial['traj']['time']
            tr = trial['crossing_times']
            
            p = vddm_params['hiker']
            vdd_taus = mangle_tau(trial['traj'], **p), mangle_tau(trial['traj_lead'], **p)
            vddm_pdf = vddm.blocker_decisions(actgrid, *vdd_taus)
            p = tdm_params['hiker']
            tdm_taus = mangle_tau(trial['traj'], **p), mangle_tau(trial['traj_lead'], **p)
            tdm_pdf = tdm.blocker_decisions(*tdm_taus, dt)
            
            allcts.append(tr[np.isfinite(tr)])
            allpreds.append(
                    scipy.interpolate.interp1d(ts, np.array(vddm_pdf.ps)*len(allcts[-1]), bounds_error=False, fill_value=(0, 0))
                    )
            allpreds_tdm.append(
                    scipy.interpolate.interp1d(ts, np.array(tdm_pdf.ps)*len(allcts[-1]), bounds_error=False, fill_value=(0, 0))
                    )

            
            cdf = ecdf(tr)(ts)
            cdf_vdd = np.cumsum(np.array(vddm_pdf.ps)*dt)
            cdf_tdm = np.cumsum(np.array(tdm_pdf.ps)*dt)

            tau = trial['traj']['tau']
            tau_b = trial['traj_lead']['tau']
            #early_t = np.min(ts[(tau_b > 0) & (tau <= 0)], initial=np.inf)
            #early_t = min(trial['t_stop'], early_t, ts[-1])
            #early_i = np.searchsorted(ts, early_t)
            stats.append(dict(
                mean=np.mean(tr[np.isfinite(tr)]),
                mean_vdd=np.dot(np.array(vddm_pdf.ps)*dt/(1 - vddm_pdf.uncrossed), ts),
                mean_tdm=np.dot(np.array(tdm_pdf.ps)*dt/(1 - tdm_pdf.uncrossed), ts),
                has_accel=np.std(trial['traj']['speed']) > 0.01
                ))
            
            d0 = -scipy.interpolate.interp1d(ts, trial['traj']['distance'])(0)
            plt.plot(ts, ecdf(tr)(ts)*100, label=f'Empirical, d0={d0:.1f} m, ehmi={trial["has_hmi"]}')
        
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
    rng = stats['mean'].min(), stats['mean'].max()
    plt.plot(rng, rng, 'C0-')
    
    plt.plot(stats['mean'], stats.mean_vdd, 'ko', label='VDDM')
    plt.plot(stats['mean'], stats.mean_tdm, 'kx', label='TDM')
    
    #vstats = stats[~stats.has_accel]
    #plt.plot(vstats['mean'], vstats.mean_vdd, 'C0o', label='VDDM (constant speed)')
    #plt.plot(vstats['mean'], vstats.mean_tdm, 'C1o', label='TDM (constant speed)')
    #vstats = stats[stats.has_accel]
    #plt.plot(vstats['mean'], vstats.mean_vdd, 'C0x', label='VDDM (variable speed)')
    #plt.plot(vstats['mean'], vstats.mean_tdm, 'C1x', label='TDM (variable speed)')
    
    plt.legend()
    plt.xlabel('Measured mean crossing time (seconds)')
    plt.ylabel('Predicted mean crossing time (seconds)')
    plt.axis('equal')
    plt.show()
    
    allcts = np.concatenate(allcts)
    allt = (np.concatenate([i.x for i in allpreds]))
    rng = np.arange(np.min(allcts), np.max(allcts), 0.05)
    pred = np.sum([i(rng) for i in allpreds], axis=0)/len(allcts)
    pred_tdm = np.sum([i(rng) for i in allpreds_tdm], axis=0)/len(allcts)
    plt.plot(rng, pred, color='black', label='VDDM')
    plt.plot(rng, pred_tdm, '--', color='black', label='TDM')

    rng = np.arange(rng[0], rng[-1], 0.2)
    plt.hist(allcts, bins=rng, density=True, color='C0', label='Observed')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Crossing probability density')
    plt.show()
    pdf.close()
    
DT = 1/30
leader_start = 100
vehicle_length = 0
def get_trajectory(time_gap, speed, is_braking, has_hmi, **kwargs):
    starttime = -leader_start/speed
    endtime = starttime + 20
    if not is_braking:
        endtime = time_gap

    ts = np.arange(starttime, endtime, DT)
    lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking, **kwargs)

    
    tau_lag = -lag_x/lag_speed
    tau_lag[~np.isfinite(tau_lag)] = 1e9
    lead_dist = leader_start - (ts - starttime)*speed + vehicle_length
    tau_lead = lead_dist/speed
    tau_lead[~np.isfinite(tau_lead)] = 1e9
    
    lead_traj = np.rec.fromarrays(
            (ts, lead_dist, np.repeat(speed, len(ts)), tau_lead, np.gradient(tau_lead, DT), np.zeros(len(ts))),
            names="time,distance,speed,tau,tau_dot,ehmi")
    
    ehmi = np.zeros(len(ts))
    if has_hmi:
        ehmi[ts >= t_brake] = 1.0
    
    traj = np.rec.fromarrays((ts, -lag_x, lag_speed, tau_lag, np.gradient(tau_lag, DT), ehmi), names="time,distance,speed,tau,tau_dot,ehmi")

    return traj, lead_traj

def plot_schematic():
    dt = 1/30
    speed = 15.6464
    time_gap = 4
    ehmi = True
    is_braking = True

    traj, traj_lead = get_trajectory(time_gap, speed, is_braking, ehmi)
    
    traj = traj[traj.time >= 0]
    traj = traj[traj.time < 8]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(12, 2)
    
    tauax = fig.add_subplot(gs[0:3,0])
    tauax.set_ylabel("TTA (s)")
    tauax.plot(traj.time, traj.tau, 'k')
    tauax.set_ylim(0, 10)
    tauax.get_xaxis().set_visible(False)
    
    taudotax = fig.add_subplot(gs[3:6,0])
    taudotax.set_ylabel("TTA change (s/s)")
    taudotax.plot(traj.time, traj.tau_dot, 'k')
    taudotax.set_ylim(-2, 2)
    taudotax.get_xaxis().set_visible(False)
    
    distax = fig.add_subplot(gs[6:9,0])
    distax.set_ylabel("Distance (m)")
    distax.plot(traj.time, traj.distance, 'k')
    distax.get_xaxis().set_visible(False)

    ehmiax = fig.add_subplot(gs[9:12,0])
    ehmiax.set_ylabel("eHMI active")
    ehmiax.plot(traj.time, traj.ehmi, 'k')
    ehmiax.set_xlabel("Time (seconds)")

   
    params = vddm_params['hiker']
    model = Vddm(dt=dt, **model_params(params))
    inp = mangle_tau(traj, **params)
    
    allweights = np.zeros((len(traj), actgrid.N))
    weights = np.zeros(actgrid.N)
    
    weights[actgrid.bin(0)] = 1.0
    undecided = 1.0
    crossed = []
    for i in range(len(traj)):
        weights, decided = model.step(actgrid, inp[i], weights, 1.0)
        undecided -= undecided*decided
        crossed.append(1 - undecided)
        allweights[i] = weights
        allweights[i] *= undecided
    
    inpax = fig.add_subplot(gs[0:4,1])
    inpax.plot(traj.time, inp, 'k')
    inpax.set_ylim(0, 10)
    inpax.set_ylabel("Observation")
    inpax.get_xaxis().set_visible(False)

    actax = fig.add_subplot(gs[4:8,1])
    actax.set_ylabel("Activation")
    actax.get_xaxis().set_visible(False)
 
    actax.pcolormesh(traj.time, np.linspace(actgrid.low(), actgrid.high(), actgrid.N),
            allweights.T/actgrid.dx,
            vmax=0.5, cmap='jet')
    #actax.plot(traj.time, inp)
    actax.set_ylim(actgrid.low(), params['act_threshold'])
    
    crossax = fig.add_subplot(gs[8:12,1])
    crossax.plot(traj.time, crossed, 'k')
    crossax.set_ylabel("Decided")
    
    crossax.set_xlabel("Time (seconds)")

    plt.show()

    


def plot_sample_trials():
    dt = 1/30
    speed = 13.4
    time_gap = 4
    
    p = vddm_params['hiker']
    vddm = Vddm(dt=dt, **model_params(p))
    def predict(traj, traj_lead):
        vdd_taus = mangle_tau(traj, **p), mangle_tau(traj_lead, **p)
        return np.array(vddm.blocker_decisions(actgrid, *vdd_taus).ps)
    
    """
    p = tdm_params['hiker']
    tdm = Tdm(**model_params(p))
    def predict(traj, traj_lead):
        tdm_taus = mangle_tau(traj, **p), mangle_tau(traj_lead, **p)
        return np.array(tdm.blocker_decisions(*tdm_taus, dt=dt).ps)
    """
    
    data = pd.read_csv('hiker_cts.csv')
    data = data.query('braking_condition <= 3')
    
    nd = data.query("not is_braking and time_gap == @time_gap")
    for i, (s, sd) in enumerate(nd.groupby('speed')):
        #cartime = vehicle_length/s
        cartime = 0
        traj, traj_b = get_trajectory(time_gap, s, False, False)
        emp = ecdf(sd.crossing_time)
        fit_pdf = predict(traj, traj_b)
        fit_cdf = np.cumsum(fit_pdf*dt)
        
        color = f"C{i}"
        d0 = s*time_gap
        plt.plot(traj.time + cartime, emp(traj.time), color=color, label=f"Initial distance {round(d0, 1)} m")
        #bins = np.arange(traj.time[0], traj.time[-1], 0.1)
        #plt.hist(sd.crossing_time + cartime, histtype='step', density=True, bins=bins)
        plt.plot(traj.time + cartime, fit_cdf, '--', color=color)
    
    plt.xlabel("Time since first car crossing (seconds)")
    plt.ylabel("Share crossed")
    plt.legend()
    plt.xlim(-1, 4)
    plt.show()
    
    nd = data.query("is_braking and time_gap == @time_gap and abs(speed - @speed) < 0.1")
    for i, (ehmi, sd) in enumerate(nd.groupby('has_ehmi')):
        cartime = 0
        traj, traj_b = get_trajectory(time_gap, speed, True, ehmi)
        emp = ecdf(sd.crossing_time)
        fit_pdf = predict(traj, traj_b)
        fit_cdf = np.cumsum(fit_pdf*dt)
        
        color = f"C{i}"
        label = ["Without eHMI", "With eHMI"][bool(ehmi)]
        plt.plot(traj.time + cartime, emp(traj.time), color=color, label=label)
        #bins = np.arange(traj.time[0], traj.time[-1], 0.1)
        #plt.hist(sd.crossing_time + cartime, histtype='step', density=True, bins=bins)
        plt.plot(traj.time + cartime, fit_cdf, '--', color=color)
    
    plt.xlabel("Time since first car crossing (seconds)")
    plt.ylabel("Share crossed")
    plt.legend()
    plt.xlim(-1, 12)
    plt.show()
    

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

    #plot_sample_trials()
    #plot_schematic()
