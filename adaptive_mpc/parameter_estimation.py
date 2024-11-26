import numpy as np

from vehicle_model import vehicle_dynamics

from scipy.integrate import odeint

m=2.923
Iz=0.0796
lf=0.168
lr=0.163

P = np.eye(2) * 5  # Example covariance matrix

# Pf_actual_k = 2  # Actual front lateral force at time k
# Pr_actual_k = 2  # Actual rear lateral force at time k

Pf_actual_k = 5  # Actual front lateral force at time k
Pr_actual_k = 5  # Actual rear lateral force at time k

def parameter_estimate(current_state, control_input, dt, theta):
    # Extract measured outputs (vy and omega)
    Fi_k = control_input[0]  # Input force at time k
    delta_k = control_input[1]  # Steering angle at time k

    # Simulate vehicle dynamics for one step using current parameter estimates
    state_next, alpha_f_k, alpha_r_k = vehicle_dynamics(current_state, dt, m, Iz, lf, lr, Pf_actual_k, Pr_actual_k, Fi_k, delta_k)
    
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
    new_theta = theta + 0.05 * theta_dot

    return new_theta, state_next