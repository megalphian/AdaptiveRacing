import numpy as np

def vehicle_dynamics(state, dt, m, Iz, lf, lr, Pf, Pr, Fi, delta):
    x, y, phi, vx, vy, omega = state

    Fi_temp = Fi
    if(Fi < 0 and vx < 0.01):
        Fi_temp = 0
    
    # Slip angles calculation
    if(vx <= 1):
        alpha_f = 0
        alpha_r = 0
    else:
        alpha_f = -np.arctan((vy + lf * omega) / (vx)) + delta
        alpha_r = np.arctan((-vy + lr * omega) / (vx))

    # Tire forces using Simplified Pacejka model
    Ff = Pf * alpha_f  # Front lateral force
    Fr = Pr * alpha_r  # Rear lateral force

    Ff_long = Fi_temp * np.cos(delta)

    # Vehicle dynamics equations
    dx = vx * np.cos(phi) - vy * np.sin(phi)
    dy = vx * np.sin(phi) + vy * np.cos(phi)
    dphi = omega
    dvx = (1 / m) * (Fi_temp + Ff_long * np.cos(delta) -Ff * np.sin(delta) + m * vy * omega)
    dvy = (1 / m) * (Ff_long * np.sin(delta) + Ff * np.cos(delta) + Fr - m * vx * omega)
    domega = (1 / Iz) * (Ff_long * lf * np.sin(delta) + Ff * lf * np.cos(delta) - Fr * lr)

    # Return state derivatives
    dstate = np.array([dx, dy, dphi, dvx, dvy, domega])

    new_state = state + dstate*dt
    return new_state