import numpy as np

from vehicle_model import vehicle_dynamics

from scipy.integrate import odeint

m = 1500  # Example mass
Iz = 3000  # Example moment of inertia
lf = 1.2  # Example distance from CG to front axle
lr = 1.6  # Example distance from CG to rear axle

P = np.eye(2) * 5  # Example covariance matrix

Pf_actual_k = 0.85  # Actual front lateral force at time k
Pr_actual_k = 0.55  # Actual rear lateral force at time k

def parameter_estimate(current_state, control_input, dt, theta):
    # Extract measured outputs (vy and omega)
    vx_k = current_state[3]  # Longitudinal velocity
    vy_k = current_state[4]  # Lateral velocity
    omega_k = current_state[5]  # Yaw rate

    Fi_k = control_input[0]  # Input force at time k
    delta_k = control_input[1]  # Steering angle at time k

    # Simulate vehicle dynamics for one step using current parameter estimates
    state_next = vehicle_dynamics(current_state, dt, m, Iz, lf, lr, Pf_actual_k, Pr_actual_k, Fi_k, delta_k)
    
    if vx_k <= 0.01:
        alpha_f_k = 0
        alpha_r_k = 0
        print('Vehicle is not moving')
    else:
        # Calculate the slip angles based on current state
        alpha_f_k = -np.arctan((vy_k + lf * omega_k) / vx_k) + delta_k
        alpha_r_k = np.arctan((-vy_k + lr * omega_k) / vx_k)
    
    # Construct the regressor vector (phi_k)
    phi_k = np.array([alpha_f_k, alpha_r_k])
    
    # Predicted lateral forces based on current parameter estimates
    F_f_pred = theta[0] * alpha_f_k  # Front lateral force
    F_r_pred = theta[1] * alpha_r_k  # Rear lateral force
    
    # Assume that we can measure the force experienced by the tires
    F_f_act = Pf_actual_k * alpha_f_k # Front lateral force
    F_f_act += np.random.normal(0, abs(F_f_act*0.025))  # Add noise to the measurements
    F_r_act = Pr_actual_k * alpha_r_k # Rear lateral force
    F_r_act += np.random.normal(0, abs(F_r_act*0.025))  # Add noise to the measurements

    # Measurement (using both lateral velocity and yaw rate)
    y_meas = np.array([F_f_act, F_r_act])  # Actual measurements
    y_pred = np.array([F_f_pred, F_r_pred])  # Predicted values
    
    error = y_meas - y_pred
    theta_dot = P @ error * phi_k
    # Update parameter estimates
    new_theta = theta + 0.01 * theta_dot

    return new_theta, state_next