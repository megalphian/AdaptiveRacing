% Vehicle parameters (modify as needed)
m = 1500;         % Mass of vehicle (kg)
Iz = 3000;        % Yaw moment of inertia (kg.m^2)
lf = 1.1;         % Distance from CG to front axle (m)
lr = 1.6;         % Distance from CG to rear axle (m)

% Initial guesses for unknown parameters
Pf_initial = 40000;  % Initial guess for front tire lateral stiffness
Pr_initial = 30000;  % Initial guess for rear tire lateral stiffness

Pf_actual = 60000;
Pr_actual = 40000;
% Steering angle and velocity (constant for now)
delta = 0.01;      % Steering angle (rad)
vx = 15;           % Longitudinal velocity (m/s)

% Simulation time span
dt = 0.01;         % Time step (seconds)
tspan = 0:dt:1000;   % Time vector from 0 to 10 seconds

% Initial state
initial_state = [0; 0; 0; vx; 0; 0];  % [x, y, phi, vx, vy, omega]

theta = [Pf_initial; Pr_initial];  % Initial guesses for Pf and Pr
P = eye(2)* 20;                 % Initial covariance matrix (large uncertainty)
lambda = 0.99;                     % Forgetting factor (close to 1)

% Preallocate arrays to store the parameter estimates
Pf_est = zeros(length(tspan), 1);
Pr_est = zeros(length(tspan), 1);

% Store initial estimates
Pf_est(1) = Pf_initial;  
Pr_est(1) = Pr_initial;  

% Simulate and estimate parameters at each time step
for k = 1:length(tspan)-1
    % Get the current time and state
    t = tspan(k);
    
    % Varying steering input (sinusoidal for more excitation)
    delta = 0.05;  % Updated delta calculation

    % Simulate vehicle dynamics for one step using current parameter estimates
    [~, state_next] = ode45(@(t, state) vehicle_dynamics(t, state, m, Iz, lf, lr, Pf_actual, Pr_actual, delta), ...
                             [tspan(k), tspan(k+1)], initial_state);
    state_next = state_next(end, :);  % Get the final state from ode45 result
    
    % Update the state for the next iteration
    initial_state = state_next; 

    % Extract measured outputs (vy and omega)
    vy_k = state_next(5);   % Lateral velocity
    omega_k = state_next(6); % Yaw rate
    
    % Calculate the slip angles based on current state
    alpha_f_k = -atan((vy_k + lf * omega_k) / vx) + delta;
    alpha_r_k = atan((-vy_k + lr * omega_k) / vx);
    
    % Construct the regressor vector (phi_k)
    phi_k = [alpha_f_k; alpha_r_k];
    
    % Predicted lateral forces based on current parameter estimates
    F_f_pred = theta(1) * alpha_f_k;  % Front lateral force
    F_r_pred = theta(2) * alpha_r_k;  % Rear lateral force

    F_f_act = Pf_actual * alpha_f_k; 
    F_r_act = Pr_actual * alpha_r_k; 
    
    % Predict vy and omega using the dynamics equations
    vy_pred = (1/m) * (F_f_pred * cos(delta) + F_r_pred - m * vx * omega_k);
    omega_pred = (1/Iz) * (F_f_pred * lf * cos(delta) - F_r_pred * lr);
    
    % Measurement (using both lateral velocity and yaw rate)
    y_meas = [F_f_act; F_r_act];  % Actual measurements
    y_pred = [F_f_pred; F_r_pred];  % Predicted values
    
    error = y_meas - y_pred;
    theta_dot = P * error .* phi_k;    
    % Update parameter estimates
    theta = theta + 0.01 * (theta_dot);      
    % Store the parameter estimates
    Pf_est(k+1) = theta(1);
    Pr_est(k+1) = theta(2);
end

% Plot the results
figure;
subplot(2,1,1);
plot(tspan, Pf_est);
title('Estimated Pf over Time');
xlabel('Time (s)');
ylabel('Pf (N/rad)');

subplot(2,1,2);
plot(tspan, Pr_est);
title('Estimated Pr over Time');
xlabel('Time (s)');
ylabel('Pr (N/rad)');

function dstate = vehicle_dynamics(t, state, m, Iz, lf, lr, Pf, Pr, delta)

    % State variables
    x = state(1);     % Position x
    y = state(2);     % Position y
    phi = state(3);   % Yaw angle
    vx = state(4);    % Longitudinal velocity
    vy = state(5);    % Lateral velocity
    omega = state(6); % Yaw rate

    % Slip angles calculation
    alpha_f = -atan((vy + lf * omega) / vx) + delta;
    alpha_r = atan((-vy + lr * omega) / vx);

    % Tire forces using Simplified Pacejka model
    Ff = Pf * alpha_f; % Front lateral force
    Fr = Pr * alpha_r; % Rear lateral force

    % Vehicle dynamics equations
    dx = vx * cos(phi) - vy * sin(phi);
    dy = vx * sin(phi) + vy * cos(phi);
    dphi = omega;
    dvx = (1 / m) * (-Ff * sin(delta) + m * vy * omega);
    dvy = (1 / m) * (Ff * cos(delta) + Fr - m * vx * omega);
    domega = (1 / Iz) * (Ff * lf * cos(delta) - Fr * lr);

    % Return state derivatives
    dstate = [dx; dy; dphi; dvx; dvy; domega];
end
