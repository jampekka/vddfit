import numpy as np

x_stop = -2.5
x_brake = -38.5

def braking_spec(time_gap, speed, x_stop=x_stop, x_brake=x_brake):
    if x_brake >= x_stop:
        return 0, np.inf, np.inf
    v0 = speed
    x0 = -time_gap*v0
 

    brake_dist = x_stop - x_brake
    
    a = -v0**2/(2*brake_dist)
    brake_dur = -v0/a
    
    t_brake = (x_brake - x0)/v0
    t_stop = t_brake + brake_dur

    return a, t_brake, t_stop

def simulate_trajectory(t, time_gap, speed, braking, x_stop=x_stop, x_brake=x_brake):
    v0 = speed
    x0 = -time_gap*v0
    
    if not braking:
        x = t*v0 + x0
        v = np.repeat(speed, len(t))
        return x, v, (np.inf, np.inf)
    
    a, t_brake, t_stop = braking_spec(time_gap, speed)

    v = np.zeros(len(t))

    v = np.piecewise(t,
            [(t <= t_brake), (t > t_brake) & (t <= t_stop)],
            [lambda t: v0, lambda t: (t - t_brake)*a + v0, 0.0],
            )
    
    x = np.piecewise(t,
            [(t <= t_brake), (t > t_brake) & (t <= t_stop)],
            [lambda t: (t*v0 + x0), lambda t: (t_brake*v0 + x0) + (t - t_brake)*v0 + a/2*(t - t_brake)**2, x_stop]
            )

    return x, v, (t_brake, t_stop)
    

def test():
    import matplotlib.pyplot as plt
    v0 = 25/2.237
    time_gap = 2.0
    trng = np.linspace(-3, 20, 100)
    x, v, (t_brake, t_stop) = simulate_trajectory(trng, time_gap, v0, True)

    #plt.plot(trng, v)
    #plt.plot(trng, x)
    tau = -x/v
    plt.plot(trng, tau)
    plt.axvline(0, color='black')
    plt.axhline(time_gap, color='black')
    plt.show()

if __name__ == '__main__':
    test()
