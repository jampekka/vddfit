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
import kwopt
#_minimizer = minimizer
#def minimizer(loss, **kwargs):
#    return _minimizer(loss, opt_f=scipy.optimize.basinhopping, minimizer_kwargs=kwargs)


DT = 1/30
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

    #'hiker': {'std': 0.6384974134032458, 'damping': 0.10186865847575911, 'scale': 0.2791830034375026, 'tau_threshold': 5.544952926633872, 'act_threshold': 1.4194944054263028, 'pass_threshold': 1.6722899150569444, 'dot_coeff': 5.5075638112727745, 'ehmi_coeff': 0.8482795240546034, 'dist_coeff': 0.4016305596474629}

    #'hiker': {'std': 0.6368401155044189, 'damping': 0.10504380918528354, 'scale': 0.2718230151392075, 'tau_threshold': 5.509627698960811, 'act_threshold': 1.4199126717680992, 'pass_threshold': 1.6722891886738138, 'dot_coeff': 5.5799566165845, 'ehmi_coeff': 0.47724126592004795, 'dist_coeff': 0.40163262102258657},

    # A first quick powell fit from the previous hiker values
    #'unified': {'std': 0.6558547131963791, 'damping': 0.09913879511497192, 'scale': 0.27224618351053537, 'tau_threshold': 5.517019096527725, 'act_threshold': 1.4436305690444342, 'pass_threshold': 1.6721177255112174, 'dot_coeff': 5.543668277443149, 'ehmi_coeff': 0.5304533310844964, 'dist_coeff': 0.39950975746351836},
    #'unified': {'std': 0.6747077314882771, 'damping': 1.4911827961638806, 'scale': 1.0605907628525664, 'tau_threshold': 2.561950744209917, 'act_threshold': 0.989999457038032, 'pass_threshold': 0.7177158069371236, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.22562395565228743}
    #'unified': {'std': 0.6714706667949863, 'damping': 1.3297068690871066, 'scale': 0.8980605549307248, 'tau_threshold': 2.6652088316253293, 'act_threshold': 0.9900001026088022, 'pass_threshold': 0.7180952350619795, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.21214932901594483},
    #'unified': {'std': 0.6712429538925937, 'damping': 1.3260607076620856, 'scale': 0.8992113108584929, 'tau_threshold': 2.6694337770093473, 'act_threshold': 0.9900000000187759, 'pass_threshold': 0.7180583191752358, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.21230195653562328}
    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.21231749517573556},

    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2940552884232235, 'ehmi_coeff': 0, 'dist_coeff': 0.21231749517573556}
    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2940554727554976, 'ehmi_coeff': 0, 'dist_coeff': 0.21231749517573556},

    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2940554727554976, 'ehmi_coeff': 0.27190755157405394, 'dist_coeff': 0.21231749517573556},
    
    #'unified': {'std': 0.6862440192558461, 'damping': 1.223983663356895, 'scale': 0.6047489020410377, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000139199838, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2206429111708295, 'ehmi_coeff': 0.34911133536812883, 'dist_coeff': 0.21414996598613342}
    
    # Constant speed only, with fixed keio
    #'unified': {'std': 0.726497450046289, 'damping': 0.5828631499887685, 'scale': 0.5723807755997636, 'tau_threshold': 2.8905817957021087, 'act_threshold': 1.5901754927214726, 'pass_threshold': 1.3446477949026612, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.342317232657446}
    'unified': {'std': 0.7428544072653329, 'damping': 0.6370630838672563, 'scale': 0.6968811603124203, 'tau_threshold': 3.1131673216811158, 'act_threshold': 1.4700000000001503, 'pass_threshold': 1.2087810735174502, 'dot_coeff': 0, 'ehmi_coeff': 0, 'dist_coeff': 0.3461715480587001}

    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.2894392651525743, 'ehmi_coeff': 0.2676048639627061, 'dist_coeff': 0.21231749517573556}
    #'unified': {'std': 0.6710851225907791, 'damping': 1.3241814734998518, 'scale': 0.8998325180482395, 'tau_threshold': 2.671709646930018, 'act_threshold': 0.9900000000051525, 'pass_threshold': 0.7180524609680158, 'dot_coeff': 1.288879923787391, 'ehmi_coeff': 0.27558465363024653, 'dist_coeff': 0.21231749517573556},

    #'unified': {'std': 0.6603592740326082, 'damping': 1.1052455453328458, 'scale': 0.5141555441698754, 'tau_threshold': 2.681525975300402, 'act_threshold': 0.9900000000012792, 'pass_threshold': 0.7843853329736621, 'dot_coeff': 1.2279377218791985, 'ehmi_coeff': 0.373094808195673, 'dist_coeff': 0.20647930981743343}
    #'unified': {'std': 0.69746300229369, 'damping': 0.46613114379319825, 'scale': 0.3853993922254003, 'tau_threshold': 3.6592793898164855, 'act_threshold': 1.3499999966920435, 'pass_threshold': 1.31085488, 'dot_coeff': 2.68806523, 'ehmi_coeff': 0.42923702, 'dist_coeff': 0.27769681}
    }
tdm_params = {
    'keio_uk': {'thm': 1.4822542031746122, 'ths': 0.4160580398784046, 'lagm': -0.018037627273216346, 'lags': 0.5394395959794362, 'pass_th': -0.07255736056398886},
    'keio_japan': {'thm': 1.7737959051473795, 'ths': 0.4270654760140321, 'lagm': 0.15978825057281343, 'lags': 0.5678380232026445, 'pass_th': -0.015453144629448029},
    #'hiker': {'thm': 1.58413762002706, 'ths': 0.30787584232846543, 'lagm': 0.18325303499450624, 'lags': 0.24160198818000903, 'pass_th': 0.9960908329751528},
    #'hiker': {'thm': 1.5152455089318642, 'ths': 0.32602324226593116, 'lagm': 0.1245117672647987, 'lags': 0.6021881227736422, 'pass_th': 0.7477010070264796, 'dot_coeff': 2.6937678922001163, 'ehmi_coeff': 0.2805016925833088}
    #'hiker': {'thm': 1.5442566242657068, 'ths': 0.3166462662796281, 'lagm': 0.20509904820328212, 'lags': 0.5482381938105031, 'pass_th': 0.8428738656551962, 'dot_coeff': 3.207119275055747, 'ehmi_coeff': 0.36038229654487886, 'dist_coeff': 0.2233817144842581}

    'hiker': {'thm': 1.5589096041551214, 'ths': 0.31723512786804214, 'lagm': 0.22346854221177187, 'lags': 0.5274406179540562, 'pass_th': 0.8792264031648431, 'dot_coeff': 3.2118358447257203, 'ehmi_coeff': 0.91032775986075, 'dist_coeff': 0.19840058606111935},
    
    # A first quick powell fit from the previous hiker values
    'unified': {'thm': 1.5587824305997904, 'ths': 0.3271553709034499, 'lagm': 0.22576407802136994, 'lags': 0.5386053750104438, 'pass_th': 0.8792264031911448, 'dot_coeff': 3.21409347065363, 'ehmi_coeff': 0.8082487302244402, 'dist_coeff': 0.19834381246597568}
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
    
def fit_blocker_vdd(trials, dt, init=vddm_params['unified']):
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

def fit_blocker_tdm(trials, dt, init=tdm_params['unified']):
    spec = dict(
        thm=            (init['thm'],),
        ths=            (init['ths'], logbarrier),
        lagm=            (init['lagm'],),
        lags=            (init['lags'], logbarrier),
        pass_th=          (init['pass_th']),
        dot_coeff=      (init['dot_coeff'],),
        ehmi_coeff=      (init['ehmi_coeff'],),
        dist_coeff=     (init['dist_coeff'],),
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

def fit_unified_vddm(trials, dt, init=vddm_params['unified']):
    spec = dict(
        std=            (init['std'], logbarrier,),
        damping=        (init['damping'], logbarrier,),
        scale=          (init['scale'], logbarrier,),
        tau_threshold=  (init['tau_threshold'], logbarrier,fixed),
        act_threshold=  (init['act_threshold'], logbarrier,),
        pass_threshold= (init['pass_threshold'],fixed),
        dot_coeff=      (init['dot_coeff'],),
        ehmi_coeff=      (init['ehmi_coeff'],),
        dist_coeff=     (init['dist_coeff'],)
            )
    
    bestlik = -np.inf
    def loss(**params):
        lik = 0
        model = Vddm(dt=dt, **model_params(params))
        for trial in trials:
            if len(trial) == 3:
                traj, traj_b, rts = trial
                tau = mangle_tau(traj, **params)
                tau_b = mangle_tau(traj_b, **params)
                pdf = model.blocker_decisions(actgrid, tau, tau_b)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            else:
                traj, rts = trial
                tau = mangle_tau(traj, **params)
                pdf = model.decisions(actgrid, tau)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
        nonlocal bestlik
        if lik > bestlik:
            bestlik = lik
            print(lik)
        return -lik
    
    def cb(x, f, accept):
        print(kwopt.unmangle(spec, x))
        print(f, accept)
    
    return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)
    
    #return minimizer(loss, method='powell')(**spec)
    return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)
    

def fit_unified_vddm_consttau(trials, dt, init=vddm_params['unified']):
    spec = dict(
        std=            (init['std'], logbarrier),
        damping=        (init['damping'], logbarrier),
        scale=          (init['scale'], logbarrier),
        tau_threshold=  (init['tau_threshold'], logbarrier),
        act_threshold=  (init['act_threshold'], logbarrier),# (init['act_threshold'], logbarrier),
        pass_threshold= (init['pass_threshold'],),
        dot_coeff=      (0,fixed),
        ehmi_coeff=      (0,fixed),
        dist_coeff=     (init['dist_coeff'],)
    )
    
    bestlik = -np.inf
    def loss(**params):
        lik = 0
        
        model = Vddm(dt=dt, **model_params(params))
        for trial in trials:
            if len(trial) == 3:
                traj, traj_b, rts = trial
                tau = mangle_tau(traj, **params)
                tau_b = mangle_tau(traj_b, **params)
                pdf = model.blocker_decisions(actgrid, tau, tau_b)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            else:
                traj, rts = trial
                tau = mangle_tau(traj, **params)
                pdf = model.decisions(actgrid, tau)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
        nonlocal bestlik
        if lik > bestlik:
            bestlik = lik
            #print(params)
            print(lik)
        return -lik
    
    def cb(x, f, accept):
        print(kwopt.unmangle(spec, x))
        print(f, accept)
    
    return minimizer(loss, scipy.optimize.basinhopping, T=10.0,
            callback=cb, minimizer_kwargs={'method': 'powell'}
            #method='powell', #options={'maxiter': 1}
            )(**spec)
    #return minimizer(loss)(**spec)

def fit_unified_tdm(trials, dt, init=tdm_params['unified']):
    spec = dict(
        thm=            (init['thm'],),
        ths=            (init['ths'], logbarrier),
        lagm=            (init['lagm'],),
        lags=            (init['lags'], logbarrier),
        pass_th=          (init['pass_th']),
        dot_coeff=      (init['dot_coeff'],),
        ehmi_coeff=      (init['ehmi_coeff'],),
        dist_coeff=     (init['dist_coeff'],),
            )
    
    def loss(**params):
        lik = 0
        model = Tdm(**model_params(params))
        for trial in trials:
            if len(trial) == 3:
                traj, traj_b, rts = trial
                tau = mangle_tau(traj, **params)
                tau_b = mangle_tau(traj_b, **params)
                pdf = model.blocker_decisions(tau, tau_b, dt)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)
            else:
                traj, rts = trial
                tau = mangle_tau(traj, **params)
                pdf = model.decisions(tau, dt)
                lik += pdf.loglikelihood(rts - traj.time[0], np.finfo(float).eps)

        return -lik
    
    return minimizer(loss, method='powell')(**spec)

def get_hiker_trials(include_constants=True, include_decels=True, include_ehmi=True):
    data = pd.read_csv('hiker_cts.csv')
    data = data.query('braking_condition <= 3')
    if not include_constants:
        data = data.query('is_braking == True')
    if not include_decels:
        data = data.query('is_braking == False')
    if not include_ehmi:
        data = data.query('has_ehmi == False')
    
    trials = []
    for g, d in data.groupby(['time_gap', 'speed', 'is_braking', 'has_ehmi']):
        time_gap, speed, is_braking, has_hmi = g
        
        crossing_times = d.crossing_time.values
        crossing_times[~np.isfinite(crossing_times)] = np.inf

        traj, lead_traj = get_trajectory(time_gap, speed, is_braking, has_hmi)
        
        #traj = np.rec.fromarrays((-lag_x,tau_lag, np.gradient(tau_lag, DT), ehmi), names="distance,tau,tau_dot,ehmi")
        trials.append((traj, lead_traj, crossing_times))

    return trials

def get_keio_trials(country='uk', include_constants=True, include_decels=True, include_ehmi=True):
    all_trajectories = pd.read_csv('d2p_trajectories.csv').rename(columns={'time_c': 'time'})
    all_responses = pd.read_csv(f'd2p_cross_times_{country}.csv')
    all_responses[["subject_id", "trial_number"]] = all_responses.unique_ID.str.split("_", 1, expand=True)
    responses = dict(list(all_responses.groupby('trial_id')))

    trials = {}
    trial_taus = {}
    subj_trial_responses = {}
    
    all_trajectories = all_trajectories.query("trial_n > 2 and trial_n <= 16")

    for trial, traj in all_trajectories.groupby('trial_n'):
        has_decel = np.std(traj['speed']) > 0.1
        if has_decel and not include_decels:
            continue
        if not has_decel and not include_constants:
            continue
        resp = responses[trial]['cross_time'].values
        #tau = traj['tau'].values
        traj['ehmi'] = False
        # Recompute these as they are mangled in the original
        traj['tau'] = traj['distance'].values/traj['speed'].values
        traj['tau_dot'] = np.gradient(traj.tau.values, DT)
        trials[trial] = (traj.to_records(), resp)
    
    return list(trials.values())

def fit_hiker_and_keio():
    subset = dict(
        include_ehmi        = True,
        include_decels      = False,
        include_constants   = True
        )
    trials = get_hiker_trials(**subset) + get_keio_trials(**subset)

    #fit = fit_unified_tdm(trials, DT)
    #print(fit)
    
    #fit = fit_unified_vddm(trials, DT)
    fit = fit_unified_vddm_consttau(trials, DT)
    print(fit)

    
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

    vdd_fit = fit_blocker_vdd(trials, DT)
    vdd_params = vdd_fit.kwargs
    print("VDD")
    print(vdd_fit)
    
    #tdm_fit = fit_blocker_tdm(trials, DT)
    #tdm_params = tdm_fit.kwargs
    #print("TDM")
    #print(tdm_fit)


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
            # Hmm.. why only constant speed?
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
    
    vddp = vddm_params['unified']
    tdmp = tdm_params['unified']

    vddm = Vddm(dt=dt, **model_params(vddp))
    tdm = Tdm(**model_params(tdmp))

    #vddm = Vddm(dt=dt, **vddm_params[f'keio_{country}'])
    #tdm = Tdm(**tdm_params[f'keio_{country}'])
    
    pdf = PdfPages(f"keiofit_{country}.pdf")
    def show():
        pdf.savefig()
        plt.close()
    
    """
    all_trajectories = all_trajectories.query('trial_n > 2')
    tau_types = defaultdict(list)
    for trial, td in all_trajectories.groupby('trial_n'):
        has_decel = np.std(td.speed.values) > 0.01
        tau_type = trial
        if not has_decel:
            tau_type = round(td.distance.values[0]/td.speed.values[0], 1)
        
        tau_types[(tau_type, has_decel)].append((trial, td))
    """
    stats = []

    trials = get_keio_trials(country=country)
    for traj, tr in trials:
            #plt.plot(td.time_c, td.distance)
            #ts = td.time_c.values
            #tr = responses[trial]
            #tau = td.distance/td.speed
            #td['ehmi'] = False
            ts = traj.time
            tau_vddm = mangle_tau(traj, **vddp)
            vddm_pdf = vddm.decisions(actgrid, tau_vddm)
            tau_tdm = mangle_tau(traj, **tdmp)
            tdm_pdf = tdm.decisions(tau_tdm, dt)

            stats.append(dict(
                mean=np.mean(tr),
                mean_vdd=np.dot(np.array(vddm_pdf.ps)*dt/(1 - vddm_pdf.uncrossed), ts),
                mean_tdm=np.dot(np.array(tdm_pdf.ps)*dt/(1 - tdm_pdf.uncrossed), ts),
                has_accel=np.std(traj.speed) > 0.01
                ))

            label = "Empirical v0 {td.speed[0]:.1f} m/s"
            plt.plot(ts, ecdf(tr)(ts)*100, label=f'Empirical, d0={traj.distance[0]:.1f} m')
            plt.title(f"Keio {country} trial type")
            plt.plot(ts, np.cumsum(np.array(vddm_pdf.ps)*dt)*100, 'k', label='VDDM')
            plt.plot(ts, np.cumsum(np.array(tdm_pdf.ps)*dt)*100, 'k--', label='TDM')
            plt.legend()
            plt.xlim(ts[0], ts[-1])
            plt.ylim(-1, 101)
            plt.ylabel('Percentage crossed')
            plt.xlabel('Time (seconds)')
            plt.twinx()
            plt.plot(ts, traj.tau, label='Time to arrival', color='black', alpha=0.5)
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
    
    vddm = Vddm(dt=dt, **model_params(vddm_params[f'unified']))
    tdm = Tdm(**model_params(tdm_params[f'unified']))
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
            
            p = vddm_params['unified']
            vdd_taus = mangle_tau(trial['traj'], **p), mangle_tau(trial['traj_lead'], **p)
            vddm_pdf = vddm.blocker_decisions(actgrid, *vdd_taus)
            p = tdm_params['unified']
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
def get_trajectory(time_gap, speed, is_braking, has_hmi, duration=20, end_at_passed=True, **kwargs):
    starttime = -leader_start/speed
    endtime = starttime + duration
    if not is_braking and end_at_passed:
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

def plot_schematic_old_wtf():
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

   
    params = vddm_params['unified']
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

dt = 1/30
def plot_schematic():
    dt = 1/30
    speed = 15.6464
    time_gap = 4
    ehmi = True
    is_braking = False
    traj, traj_lead = get_trajectory(time_gap, speed, is_braking, ehmi, end_at_passed=False)
    plot_traj_schematic(traj)

def plot_keio_schematics():
    trials = get_keio_trials()
    for traj, _ in trials:
        plot_traj_schematic(traj)

def plot_traj_schematic(traj):
    
    traj = traj[traj.time >= 0]
    #traj = traj[traj.time < 8]

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

   
    params = vddm_params['unified']
    model = Vddm(dt=dt, **model_params(params))
    inp = mangle_tau(traj, **params)
    
    tparams = tdm_params['unified']
    tdm = Tdm(**model_params(tparams))
    tinp = mangle_tau(traj, **tparams)
    
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
    
    inpax = fig.add_subplot(gs[0:3,1])
    inpax.plot(traj.time, inp, 'k')
    inpax.plot(traj.time, tinp, 'k--')
    inpax.set_ylim(0, 10)
    inpax.set_ylabel("Observation")
    inpax.get_xaxis().set_visible(False)

    actax = fig.add_subplot(gs[3:6,1])
    actax.set_ylabel("Activation")
    actax.get_xaxis().set_visible(False)
 
    actax.pcolormesh(traj.time, np.linspace(actgrid.low(), actgrid.high(), actgrid.N),
            allweights.T/actgrid.dx,
            vmax=0.5, cmap='jet')
    #actax.plot(traj.time, inp)
    actax.set_ylim(actgrid.low(), params['act_threshold'])

    tdm_pdf = np.array(tdm.decisions(tinp, dt).ps)
    tdm_crossed = np.cumsum(tdm_pdf*dt)
    
    crossax = fig.add_subplot(gs[6:9,1])
    crossax.plot(traj.time, crossed, 'k')
    crossax.plot(traj.time, tdm_crossed, 'k--')
    crossax.set_ylabel("Decided")
    
    crossax.set_xlabel("Time (seconds)")
    
    pdfax = fig.add_subplot(gs[9:12,1])
    pdfax.plot(traj.time, model.decisions(actgrid, inp).ps, 'k')
    pdfax.plot(traj.time, tdm_pdf, 'k--')
    pdfax.set_ylabel("Decision pdf")
    
    pdfax.set_xlabel("Time (seconds)")
    
    fig.align_labels()
    plt.show()


def plot_sample_trials():
    dt = 1/30
    speed = 13.4
    time_gap = 4
    
    p = vddm_params['unified']
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
    #plot_hiker()
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
    #plot_keio_schematics()
    fit_hiker_and_keio()
