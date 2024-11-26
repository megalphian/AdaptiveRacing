import numpy as np

from car_model import vehicle_dynamics

from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Simulate and estimate parameters at each time step
dt = 0.01
tspan = np.linspace(0, 200, int(200/dt))  # Time span for simulation
initial_state = np.zeros(6)  # Example initial state
initial_state[3] = 10  # Example initial longitudinal velocity
m = 1500  # Example mass
Iz = 3000  # Example moment of inertia
lf = 1.2  # Example distance from CG to front axle
lr = 1.6  # Example distance from CG to rear axle

Pf_actual = 5000  # Example actual front lateral force parameter
Pr_actual = 6000  # Example actual rear lateral force parameter

Pf_actual = np.zeros(len(tspan))
Pf_actual[:int(100/dt)] = 5000
Pf_actual[int(100/dt):] = 2000

Pr_actual = np.zeros(len(tspan))
Pr_actual[:int(100/dt)] = 6000
Pr_actual[int(100/dt):] = 3000

Pf_init = 100  # Initial guess for Pf
Pr_init = 500  # Initial guess for Pr

Fi = np.zeros(len(tspan))  # Example input force (constant)
# Create a step signal from 1000 to -1000 N at time 50s
Fi[:int(50/dt)] = 1000
Fi[int(50/dt):int(100/dt)] = -10
Fi[int(100/dt):] = 1000

delta = np.zeros(len(tspan))  # Example steering angle (constant)
delta[:int(50/dt)] = 0.4  # Constant steering angle
delta[int(50/dt):int(100/dt)] = 0.69  
delta[int(100/dt):int(110/dt)] = -0.4
delta[int(110/dt):] = 0.02

# raise Exception('Stop here')

theta = np.array([Pf_init, Pr_init])  # Initial parameter estimates
P = np.eye(2) * 5  # Example covariance matrix

Pf_est = np.zeros(len(tspan))
Pr_est = np.zeros(len(tspan))

all_y_pred = np.zeros((len(tspan), 2))
all_y_act = np.zeros((len(tspan), 2))

current_state = initial_state

for k in range(len(tspan) - 1):
    t = tspan[k]

    # Extract measured outputs (vy and omega)
    vx_k = current_state[3]  # Longitudinal velocity
    vy_k = current_state[4]  # Lateral velocity
    omega_k = current_state[5]  # Yaw rate

    Fi_k = Fi[k]  # Input force at time k
    delta_k = delta[k]  # Steering angle at time k

    Pf_actual_k = Pf_actual[k]  # Actual front lateral force at time k
    Pr_actual_k = Pr_actual[k]  # Actual rear lateral force at time k

    # Simulate vehicle dynamics for one step using current parameter estimates
    state_next = odeint(vehicle_dynamics, current_state, [tspan[k], tspan[k+1]], args=(m, Iz, lf, lr, Pf_actual_k, Pr_actual_k, Fi_k, delta_k))
    state_next = state_next[-1, :]  # Get the final state from odeint result
    
    # Update the state for the next iteration
    current_state = state_next
    
    if vx_k <= 0:
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
    
    # Predict vy and omega using the dynamics equations
    vy_pred = (1/m) * (F_f_pred * np.cos(delta_k) + F_r_pred - m * vx_k * omega_k)
    omega_pred = (1/Iz) * (F_f_pred * lf * np.cos(delta_k) - F_r_pred * lr)
    
    # Assume that we can measure the force experienced by the tires
    F_f_act = Pf_actual_k * alpha_f_k # Front lateral force
    # F_f_act += np.random.normal(0, abs(F_f_act*0.025))  # Add noise to the measurements
    F_r_act = Pr_actual_k * alpha_r_k # Rear lateral force
    # F_r_act += np.random.normal(0, abs(F_r_act*0.025))  # Add noise to the measurements

    # Measurement (using both lateral velocity and yaw rate)
    y_meas = np.array([F_f_act, F_r_act])  # Actual measurements
    y_pred = np.array([F_f_pred, F_r_pred])  # Predicted values

    all_y_pred[k+1] = y_pred
    all_y_act[k+1] = y_meas
    
    error = y_meas - y_pred
    theta_dot = P @ error * phi_k
    # Update parameter estimates
    theta = theta + 0.01 * theta_dot
    # Store the parameter estimates
    Pf_est[k+1] = theta[0]
    Pr_est[k+1] = theta[1]

# Plot the results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, Pf_est)
plt.title('Estimated Pf over Time')
plt.xlabel('Time (s)')
plt.ylabel('Pf (N/rad)')

plt.subplot(2, 1, 2)
plt.plot(tspan, Pr_est)
plt.title('Estimated Pr over Time')
plt.xlabel('Time (s)')
plt.ylabel('Pr (N/rad)')
plt.tight_layout()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, all_y_act[:, 0], label='Actual')
plt.plot(tspan, all_y_pred[:, 0], label='Predicted')
plt.title('Front Lateral Force')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(tspan, all_y_act[:, 1], label='Actual')
plt.plot(tspan, all_y_pred[:, 1], label='Predicted')
plt.title('Rear Lateral Force')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.tight_layout()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, Fi)
plt.title('Input Force')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')

plt.subplot(2, 1, 2)
plt.plot(tspan, delta)
plt.title('Steering Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.tight_layout()

plt.show()