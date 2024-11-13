import numpy as np

def vehicle_dynamics(state, t, m, Iz, lf, lr, Pf, Pr, Fi, delta):
    x, y, phi, vx, vy, omega = state

    # Slip angles calculation
    alpha_f = -np.arctan((vy + lf * omega) / vx) + delta
    alpha_r = np.arctan((-vy + lr * omega) / vx)

    # Tire forces using Simplified Pacejka model
    Ff = Pf * alpha_f  # Front lateral force
    Fr = Pr * alpha_r  # Rear lateral force

    # Vehicle dynamics equations
    dx = vx * np.cos(phi) - vy * np.sin(phi)
    dy = vx * np.sin(phi) + vy * np.cos(phi)
    dphi = omega
    dvx = (1 / m) * (Fi -Ff * np.sin(delta) + m * vy * omega)
    dvy = (1 / m) * (Ff * np.cos(delta) + Fr - m * vx * omega)
    domega = (1 / Iz) * (Ff * lf * np.cos(delta) - Fr * lr)

    # Return state derivatives
    dstate = np.array([dx, dy, dphi, dvx, dvy, domega])
    return dstate
