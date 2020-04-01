import itertools
import numpy as np
import matplotlib.pyplot as plt

from fit_d2p import vddm_params, Vddm, model_params, mangle_tau, actgrid
import hikersim
from hikersim import braking_spec

import scipy.optimize

#START_TIME = -3
#END_TIME = 20
leader_start = 100
DT = 1/30
def get_trajectory(time_gap, speed, is_braking, has_hmi, **kwargs):
    starttime = -leader_start/speed
    endtime = starttime + 20
    if not is_braking:
        endtime = time_gap

    ts = np.arange(starttime, endtime, DT)
    lag_x, lag_speed, (t_brake, t_stop) = hikersim.simulate_trajectory(ts, time_gap, speed, is_braking, **kwargs)

    
    tau_lag = -lag_x/lag_speed
    tau_lag[~np.isfinite(tau_lag)] = 1e6
    lead_dist = leader_start - (ts - starttime)*speed
    tau_lead = lead_dist/speed
    tau_lead[~np.isfinite(tau_lead)] = 1e6
    
    lead_traj = np.rec.fromarrays(
            (ts, lead_dist,tau_lead, np.gradient(tau_lead, DT), np.zeros(len(ts))),
            names="time,distance,tau,tau_dot,ehmi")
    
    ehmi = np.zeros(len(ts))
    if has_hmi:
        ehmi[ts >= t_brake] = 1.0
    
    traj = np.rec.fromarrays((ts, -lag_x,tau_lag, np.gradient(tau_lag, DT), ehmi), names="time,distance,tau,tau_dot,ehmi")

    return traj, lead_traj


def analyze_pedestrian_time_loss(model):
    ttas = np.linspace(2, 5, 30)
    speeds = np.array([25, 30, 35])/2.237
    
    fig, (tax, pax) = plt.subplots(nrows=2)
    for speed in speeds:
        vanillas = []
        ehmis = []
        for tta in ttas:
            traj = get_trajectory(tta, speed, True, False)
            dist = model(traj).ps
            mean_time = np.dot(traj[0].time, dist/np.sum(dist))
            vanillas.append(mean_time)
            
            traj = get_trajectory(tta, speed, True, True)
            dist = model(traj).ps
            mean_time = np.dot(traj[0].time, dist/np.sum(dist))
            ehmis.append(mean_time)

        vanillas = np.array(vanillas)
        ehmis = np.array(ehmis)
        tax.plot(ttas, vanillas - ehmis, label=f"Initial vehicle speed {speed:.1f} m/s")
        pax.plot(ttas, (1 - ehmis/vanillas)*100, label=f"Initial vehicle speed {speed:.1f} m/s")
    
    fig.suptitle("eHMI effect on pedestrian crossing duration")
    tax.set_ylabel("eHMI efficiency gain (seconds)")
    pax.set_ylabel("eHMI efficiency gain (percent)")
    pax.set_xlabel("Initial TTA (seconds)")
    tax.legend()
    plt.show()

cross_dur = 3.0
acceleration = 1.3
@np.vectorize
def yield_time_loss_wtf(target_speed, time_gap, t_cross, **kwargs):
    b, t_brake, t_stop = braking_spec(time_gap, target_speed, **kwargs)
    a = acceleration
    yield_dur = max(0, t_cross + cross_dur - t_brake)
    
    # TODO: Ugly sympy generated code
    return ((1/2)*b*t_brake**2/target_speed - b*t_brake*min(t_brake + yield_dur, t_brake - target_speed/b)/target_speed + (1/2)*b*min(t_brake + yield_dur, t_brake - target_speed/b)**2/target_speed - t_brake - yield_dur + ((-1/2*target_speed/a) if (b*yield_dur + target_speed <= 0) else (-1/2*b**2*yield_dur**2/(a*target_speed))) + min(t_brake + yield_dur, t_brake - target_speed/b))

@np.vectorize
def yield_time_loss(target_speed, time_gap, t_cross, **kwargs):
    v0 = target_speed
    b, t_brake, t_stop = braking_spec(time_gap, target_speed, **kwargs)
    assert t_brake < t_stop
    assert b <= 0
    assert target_speed > 0
    a = acceleration
    t_passed = t_cross + cross_dur
    dur_yield = t_passed - t_brake
    if dur_yield < 0:
        return 0.0
    if t_passed > t_stop:
        wtf = dur_yield + v0/(2*b) + v0/(2*a)
        return wtf
    #wtf = -(a*b*dur_yield**2 + 3*b**2*dur_yield**2 + v0*(4*b*dur_yield + v0))/(2*a*v0)
    wtf = b*dur_yield**2*(-a + b)/(2*a*v0)
    return wtf

@np.vectorize
def vehicle_time_loss(v0, t_brake, t_stop, t_cross):
    a = acceleration
    t_passed = t_cross + cross_dur
    dur_yield = t_passed - t_brake
    dur_brake = t_stop - t_brake
    b = -v0/dur_brake
    if dur_yield < 0:
        return 0.0
    if t_passed > t_stop:
        wtf = dur_yield + v0/(2*b) + v0/(2*a)
        return wtf
    wtf = b*dur_yield**2*(-a + b)/(2*a*v0)
    return wtf

def get_trajectory(speed, tta, decel, ehmi, ts):
    b = -decel
    x0 = -speed*tta
    dt = np.mean(np.diff(ts))
    v = np.maximum(0, speed + b*ts)
    x = x0 + np.cumsum(v*dt)

    tau = -x/v
    tau_dot = np.gradient(tau, dt)
    ehmi = np.repeat(ehmi, len(ts))

    return np.rec.fromarrays(
            (ts, -x, tau, tau_dot, ehmi),
            names="time,distance,tau,tau_dot,ehmi")

def tta_time_loss(predict, ts, dt, tta, speed, decel, ehmi):
    t_stop = speed/decel
    traj = get_trajectory(speed, tta, decel, ehmi, ts)
    
    cd = predict(traj)
    l = vehicle_time_loss(speed, 0.0, t_stop, traj.time)
    return np.dot(l, np.array(cd.ps)*dt), np.dot(ts - ts[0], np.array(cd.ps)*dt)

tta_time_loss = np.vectorize(tta_time_loss, excluded=(0, 1))
#linear_param = [-9.05167401e-01, -1.10384155e-05, 1.10375652e-05, 4.72265220e-01]

#linear_param = [-3.21072425, -0.2429603, 0.24295947, 3.77101283]
#linear_param = [-1.83690295,  0.4502298,  -0.28016829,  1.36037307]
linear_param = [-1.90056245, 0.79925133, -0.27979611, 1.46707382]

stop_margin = 2.5
def get_linear_decel(tta, speed, ehmi, tta_c, speed_c, ehmi_c, ic):
    d0 = tta*speed
    logdecel = tta_c*np.log(tta) + speed_c*np.log(speed) * ehmi_c*ehmi + ic
    stop_decel = speed**2/(2*(d0 - stop_margin))
    return (np.exp(logdecel) + 1)*stop_decel

def get_minimum_decel(tta, speed):
    d0 = tta*speed
    stop_decel = speed**2/(2*(d0 - stop_margin))
    return stop_decel

def fit_linear_decel(params, dt):
    model = Vddm(dt=dt, **model_params(params))
    def predict(traj):
        tau = mangle_tau(traj, **params)
        return model.decisions(actgrid, tau)
    
    ts = np.arange(0, 20, dt)
    ttas = np.linspace(2, 8, 5)
    speeds = np.linspace(10/3.6, 80/3.6, 5)
    ehmis = np.array([0.0, 1.0])
    
    def loss(args):
        losses = []
        for tta, speed, ehmi in itertools.product(ttas, speeds, ehmis):
            decel = get_linear_decel(tta, speed, ehmi, *args)
            loss = tta_time_loss(predict, ts, dt, tta, speed, decel, ehmi)
            losses.append(loss)
        print(args)
        print(np.mean(losses))
        return losses

    #fit = scipy.optimize.least_squares(loss, [0.0, 0.0, 0.0, 0.0])
    fit = scipy.optimize.minimize(lambda args: np.sum(loss(args)), linear_param, method='powell')
    print(fit)
    print(fit.x)



def fit_optimal_decel(params, dt):
    #params['pass_threshold'] = -np.inf
    model = Vddm(dt=dt, **model_params(params))
    def predict(traj):
        tau = mangle_tau(traj, **params)
        return model.decisions(actgrid, tau)
    
    ts = np.arange(0, 60, 1/30)
    ttas = np.linspace(2, 8, 5)
    #speeds = np.linspace(5, 30, 5)
    #speeds = np.linspace(30, 50, 80)
    speeds = np.array([5, 10, 15])
    def tta_time_loss(tta, speed, decel, ehmi=False):
        t_stop = speed/decel
        traj = get_trajectory(speed, tta, decel, ehmi, ts)
        
        cd = predict(traj)
        l = vehicle_time_loss(speed, 0.0, t_stop, traj.time)
        return np.dot(l, np.array(cd.ps)*dt)
    
    for si, speed in enumerate(speeds):
        overdecels = []
        eoverdecels = []
        for tta in ttas:
            ehmi = False
            def loss(overdecel):
                d0 = tta*speed
                stop_decel = speed**2/(2*(d0 - stop_margin))
                decel = stop_decel + overdecel
                return tta_time_loss(tta, speed, decel, ehmi=ehmi)
            fit = scipy.optimize.minimize(lambda x: loss(*np.exp(x)), np.log(1.0))
            fit.x = np.exp(fit.x)
            overdecels.append(fit.x[0])
            ehmi = True
            fit = scipy.optimize.minimize(lambda x: loss(*np.exp(x)), np.log(1.0))
            fit.x = np.exp(fit.x)
            eoverdecels.append(fit.x[0])
        
        overdecels = np.array(overdecels)
        d0 = ttas*speed
        stop_decels = speed**2/(2*(d0 - stop_margin))
        decels = stop_decels + overdecels
        edecels = np.array(eoverdecels) + stop_decels


        ldecels = get_linear_decel(ttas, speed, 0.0, *linear_param)
        plt.plot(ttas, decels, '--', label=f"Speed {speed:.1f} m/s", color=f'C{si}')
        plt.plot(ttas, ldecels, '--', label=f"Speed {speed:.1f} m/s", alpha=0.5, color=f'C{si}')
        
        ldecels = get_linear_decel(ttas, speed, 1.0, *linear_param)
        plt.plot(ttas, edecels, '-', label=f"Speed {speed:.1f} m/s, eHMI", color=f'C{si}')
        plt.plot(ttas, ldecels, '-', label=f"Speed {speed:.1f} m/s, eHMI", alpha=0.5, color=f'C{si}')
    
    plt.loglog()
    plt.xlabel("Initial TTA (seconds)")
    plt.ylabel("Optimal deceleration (m/s²)")
    plt.legend()
    plt.show()


def fig6(params, dt):
    ttas = np.linspace(0.01, 10, 49)
    #ttas = 1/np.linspace(1, 1/6, 20)
    overdecels = np.linspace(1e-1, 5.0, 50)
    decels = np.linspace(0.1, 10, 20)
    stop_margins = np.linspace(0.0, 30, 50)
    speed = 30/2.237
    ts = np.arange(0, 20, 1/30)
    res = []
    
    #params['pass_threshold'] = -np.inf
    model = Vddm(dt=dt, **model_params(params))
    def predict(traj):
        tau = mangle_tau(traj, **params)
        return model.decisions(actgrid, tau)

    def stopping_margin_to_decel(tta, margin):
        d0 = tta*speed
        decel_stop = speed**2/(2*(d0 - margin))
        return decel_stop

    def decel_to_stopping_margin(tta, decel):
        d0 = tta*speed
        return d0 - speed**2/(2*decel)

    losses = []
    margin = 0.0
    #for overdecel in overdecels:
    #
    for margin in margins:
    #for decel in decels:
        for tta in ttas:
            d0 = tta*speed
            decel_stop = speed**2/(2*(d0 - margin))
            decel = decel_stop
            #decel = overdecel + decel_stop
            if decel < decel_stop:
                losses.append(np.nan)
                continue
            t_stop = speed/decel
            #x_stop = -tta*speed + speed*t_stop - t_stop**2*decel/2
            
            ehmi = False
            traj = get_trajectory(speed, tta, decel, ehmi, ts)
            cd = predict(traj)
            l = vehicle_time_loss(speed, 0.0, t_stop, traj.time)
            
            #plt.plot(ts, cd.ps)
            #plt.twinx()
            #plt.plot(ts, traj.tau, color='black')
            #plt.plot(ts, l, color='red')
            #plt.ylim(0, 10)
            #plt.show()
            losses.append(np.dot(l, np.array(cd.ps)*dt))
            #losses.append(tta)
    
    
    y = margins
    X, Y = np.meshgrid(ttas, y)
    losses = np.array(losses).reshape(X.shape)
    #D = speed**2/(2*(tta*speed))*Y
    #losses /= losses[0]
    decel_stop = speed**2/(2*(speed*ttas - margin))
    best = np.nanargmin(losses, axis=0)
    plt.pcolor(X, Y, losses, cmap='jet')
    plt.xlabel("TTA (seconds)")
    plt.ylabel("Acceleration over minimum (m/s²)")
    plt.plot(ttas, y[best], color='black', label='Optimal constant deceleration')
    #plt.loglog()
    #plt.plot(ttas, decel_stop, color='white')
    #plt.ylabel("Deceleration (m/s²)")
    #for decel in [2.0, 3.0, 4.0, 5.0, 6.0]:
    #    plt.plot(ttas, od, color='white', alpha=0.7)
    #plt.xlim(ttas[0], ttas[-1])
    #plt.ylim(overdecels[0], overdecels[-1])
    #plt.colorbar(label="Mean time loss (seconds)")
    plt.colorbar(label="Mean time loss (seconds)")
    plt.show()
    
def analyze_vehicle_time_loss(model):
    ttas = np.linspace(2, 5, 30)
    speeds = np.array([25, 30, 35])/2.237
    
    fig, (tax, pax) = plt.subplots(nrows=2)
    for speed in speeds:
        vanillas = []
        ehmis = []
        for tta in ttas:
            traj = get_trajectory(tta, speed, True, False)
            dist = model(traj).ps
            time_losses = yield_time_loss(speed, tta, traj[0].time)
            assert np.all(time_losses >= 0)
            mean_loss = np.dot(time_losses, dist/np.sum(dist))
            vanillas.append(mean_loss)
            
            traj = get_trajectory(tta, speed, True, True)
            dist = model(traj).ps
            time_losses = yield_time_loss(speed, tta, traj[0].time)
            #assert np.all(time_losses >= 0)
            mean_loss = np.dot(time_losses, dist/np.sum(dist))
            ehmis.append(mean_loss)

        vanillas = np.array(vanillas)
        ehmis = np.array(ehmis)
        tax.plot(ttas, vanillas - ehmis, label=f"Initial vehicle speed {speed:.1f} m/s")
        pax.plot(ttas, (1 - ehmis/vanillas)*100, label=f"Initial vehicle speed {speed:.1f} m/s")
    
    fig.suptitle("eHMI effect on vehicle crossing duration")
    tax.set_ylabel("eHMI efficiency gain (seconds)")
    pax.set_ylabel("eHMI efficiency gain (percent)")
    pax.set_xlabel("Initial TTA (seconds)")
    tax.legend()
    plt.show()

def analyze_decels(model):
    ttas = np.linspace(2, 5, 5)
    speeds = np.array([25, 30, 35])/2.237
    tta = 5.0
    init_distances = np.linspace(30, 100, 5)
    x_stop = hikersim.x_stop
    speed = speeds[1]
    #for speed in speeds:
    for init_distance in init_distances:
        losses = []
        bds = np.linspace(-x_stop + 0.1, 60, 30)
        for bd in bds:
            tta = init_distance/speed
            braking = bd > x_stop
            traj = get_trajectory(tta, speed, braking, False, x_brake=-bd)
            dist = model(traj).ps
            time_losses = yield_time_loss(speed, tta, traj[0].time, x_brake=-bd)
            mean_loss = np.dot(time_losses, dist/np.sum(dist))
            losses.append(mean_loss)
            
        #plt.plot(bds, losses, label=f"Initial vehicle speed {speed:.1f} m/s")
        losses = np.array(losses)
        losses -= losses[0]
        plt.plot(bds, losses, label=f"Distance where seen {init_distance:.1f} m")
    
    plt.suptitle(f"Braking start distance effect on vehicle time loss (init speed {speed:.1f} m/s)")
    plt.gca().set_ylabel("Vehicle time loss (seconds)")
    plt.gca().set_xlabel("Braking initiation (meters)")
    plt.legend()
    plt.show()


def vdd_predictor(params, dt):
    model = Vddm(dt=dt, **model_params(params))
    def predict(traj):
        ta = mangle_tau(traj[0], **params)
        tb = mangle_tau(traj[1], **params)
        return model.blocker_decisions(actgrid, ta, tb)
    return predict

def plot_optimized_decels():
    speeds = [5, 10, 15]
    ttas = np.linspace(2, 8, 100)
    
    plt.axhline(3.5, linestyle="dashed", color='black', alpha=0.5)
    for i, speed in enumerate(speeds):
        decels_o = get_linear_decel(ttas, speed, 0, *linear_param)
        decels_oe = get_linear_decel(ttas, speed, 1, *linear_param)
        decels_min = get_minimum_decel(ttas, speed)
        color = f"C{i}"
        plt.plot(ttas, decels_o, color=color, label=f"Optimized decel, speed {speed} m/s")
        plt.plot(ttas, decels_oe, '--', color=color, label=f"Optimized decel w/ eHMI, speed {speed} m/s")
        plt.plot(ttas, decels_min, ':', color=color, label=f"Minimum decel, speed {speed} m/s")

    plt.xlim(ttas[0], ttas[-1])
    #plt.loglog()
    plt.xlabel("Initial TTA (seconds)")
    plt.ylabel("Deceleration (m/s²)")
    plt.legend()
    plt.show()

def vehicle_time_savings(params, dt):
    model = Vddm(dt=dt, **model_params(params))
    def predict(traj):
        tau = mangle_tau(traj, **params)
        return model.decisions(actgrid, tau)
    speeds = [5, 10, 15]
    ttas = np.linspace(2, 8, 100)
    ts = np.arange(0, 20, dt)
    for i, speed in enumerate(speeds):
        label=f'Speed {speed} m/s'
        color = f"C{i}"
        
        decels = get_minimum_decel(ttas, speed)
        losses, plosses = tta_time_loss(predict, ts, dt, ttas, speed, decels, 0)
        plt.figure("baseline")
        plt.title("Vehicle time loss with minimum constant deceleration")
        plt.plot(ttas, losses, label=label, color=color)
        plt.ylabel("Mean time loss (seconds)")

        plt.figure("baseline_ped")
        plt.title("Pedestrian time loss with minimum constant deceleration")
        plt.plot(ttas, plosses, label=label, color=color)
        plt.ylabel("Mean time loss (seconds)")

        
        losses_e, plosses_e = tta_time_loss(predict, ts, dt, ttas, speed, decels, 1)
        plt.figure("esave")
        plt.title("Vehicle time saving with eHMI")
        plt.plot(ttas, losses - losses_e, label=label, color=color)
        plt.ylabel("Mean time loss reduction (seconds)")

        plt.figure("esave_ped")
        plt.title("Pedestrian time saving with eHMI")
        plt.plot(ttas, plosses - plosses_e, label=label, color=color)
        plt.ylabel("Mean time loss reduction (seconds)")
        
        decels_o = get_linear_decel(ttas, speed, 0, *linear_param)
        losses_o, plosses_o = tta_time_loss(predict, ts, dt, ttas, speed, decels_o, 0)
        
        plt.figure("osave")
        plt.title("Vehicle time saving with optimized deceleration")
        plt.plot(ttas, losses - losses_o, label=label, color=color)
        plt.ylabel("Mean time loss reduction (seconds)")

        plt.figure("osave_ped")
        plt.title("Pedestrian time saving with optimized deceleration")
        plt.plot(ttas, plosses - plosses_o, label=label, color=color)
        plt.ylabel("Mean time loss reduction (seconds)")

        decels_eo = get_linear_decel(ttas, speed, 1, *linear_param)
        losses_eo, plosses_eo = tta_time_loss(predict, ts, dt, ttas, speed, decels_eo, 1)
        
        plt.figure("eosave")
        plt.title("Vehicle time saving with optimized deceleration and eHMI")
        plt.plot(ttas, losses - losses_eo, label=label, color=color)
        plt.ylabel("Mean time loss reduction (seconds)")

        plt.figure("eosave_ped")
        plt.title("Pedestrian time saving with optimized deceleration and eHMI")
        plt.plot(ttas, plosses - plosses_eo, label=label, color=color)
        plt.ylabel("Mean time loss reduction (seconds)")

    for lbl in plt.get_figlabels():
        plt.figure(lbl)
        plt.xlabel("Initial TTA (seconds)")
        plt.xlim(ttas[0], ttas[-1])
        plt.savefig(f"figs/{lbl}.svg")
    plt.show()
    """
    plt.xlim(ttas[0], ttas[-1])
    plt.xlabel("Initial TTA (seconds)")
    #plt.ylabel("Vehicle time loss (seconds)")
    plt.ylabel("Extra deceleration + eHMI time saving (seconds)")
    #plt.ylabel("eHMI extra time saving over opt decel (seconds)")
    plt.legend()
    plt.show()
    """


if __name__ == '__main__':
    pred = vdd_predictor(vddm_params['hiker'], 1/30)
    #analyze_pedestrian_time_loss(pred)
    #analyze_vehicle_time_loss(pred)
    #analyze_decels(pred)
    #fig6(vddm_params['hiker'], 1/30)
    #fit_optimal_decel(vddm_params['hiker'], 1/30)
    #fit_linear_decel(vddm_params['hiker'], 1/30)
    #vehicle_time_savings(vddm_params['hiker'], 1/30)
    plot_optimized_decels()
