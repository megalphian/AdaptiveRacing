% Vehicle parameters (example values, modify as needed)
m = 1500;         % Mass of vehicle (kg)
Iz = 3000;        % Yaw moment of inertia (kg.m^2)
lf = 1.1;         % Distance from CG to front axle (m)
lr = 1.6;         % Distance from CG to rear axle (m)

% Simplified Pacejka model parameters (example values)
Pf = 10000;       % Front tire lateral stiffness
Pr = 20000;       % Rear tire lateral stiffness

% Simulation parameters
delta = 0.05;     % Steering angle (rad)
vx = 15;          % Longitudinal velocity (m/s)
tspan = [0 10];   % Simulation time span (s)
initial_state = [0; 0; 0; vx; 0; 0]; % Initial conditions [x, y, phi, vx, vy, omega]

% Solve the differential equations
[t, states] = ode45(@(t, state) vehicle_dynamics(t, state, m, Iz, lf, lr, Pf, Pr, delta), tspan, initial_state);

% Extracting the states for plotting
x = states(:, 1);    % X position
y = states(:, 2);    % Y position
phi = states(:, 3);  % Yaw angle
vx = states(:, 4);   % Longitudinal velocity
vy = states(:, 5);   % Lateral velocity
omega = states(:, 6);% Yaw rate

alpha_f = - atan((vy + lf * omega) ./ vx) + delta;
alpha_r = atan((lr * omega - vy) ./ vx);

% Plot the results
% Create figure for position
figure;
subplot(3,1,1);
plot(t, x, 'b-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('X Position (m)', 'FontWeight', 'bold', 'FontSize', 12);
title('Vehicle X Position', 'FontWeight', 'bold', 'FontSize', 14);
legend('X Position', 'Location', 'best');

subplot(3,1,2);
plot(t, y, 'r-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Y Position (m)', 'FontWeight', 'bold', 'FontSize', 12);
title('Vehicle Y Position', 'FontWeight', 'bold', 'FontSize', 14);
legend('Y Position', 'Location', 'best');

subplot(3,1,3);
plot(t, phi, 'g-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Yaw Angle (rad)', 'FontWeight', 'bold', 'FontSize', 12);
title('Vehicle Yaw Angle', 'FontWeight', 'bold', 'FontSize', 14);
legend('Yaw Angle', 'Location', 'best');

% Create figure for velocities
figure;
subplot(2,1,1);
plot(t, vx, 'm-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Longitudinal Velocity (m/s)', 'FontWeight', 'bold', 'FontSize', 12);
title('Vehicle Longitudinal Velocity', 'FontWeight', 'bold', 'FontSize', 14);
legend('Longitudinal Velocity', 'Location', 'best');

subplot(2,1,2);
plot(t, vy, 'c-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Lateral Velocity (m/s)', 'FontWeight', 'bold', 'FontSize', 12);
title('Vehicle Lateral Velocity', 'FontWeight', 'bold', 'FontSize', 14);
legend('Lateral Velocity', 'Location', 'best');

% Create figure for yaw rate
figure;
plot(t, omega, 'k-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Yaw Rate (rad/s)', 'FontWeight', 'bold', 'FontSize', 12);
title('Vehicle Yaw Rate', 'FontWeight', 'bold', 'FontSize', 14);
legend('Yaw Rate', 'Location', 'best');

% Create figure for tire slip
figure;
subplot(2,1,1);
plot(t, alpha_f, 'm-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Forward Slip Angle (rad)', 'FontWeight', 'bold', 'FontSize', 12);
title('Forward Slip Angle', 'FontWeight', 'bold', 'FontSize', 14);
legend('Forward Slip Angle', 'Location', 'best');

subplot(2,1,2);
plot(t, alpha_r, 'c-', 'LineWidth', 2); % Improved line style
grid on; % Add grid
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Rear Slip Angle (rad)', 'FontWeight', 'bold', 'FontSize', 12);
title('Rear Slip Angle', 'FontWeight', 'bold', 'FontSize', 14);
legend('Rear Slip Angle', 'Location', 'best');


% Vehicle dynamics function
function dstate = vehicle_dynamics(t, state, m, Iz, lf, lr, Pf, Pr, delta)

    % State variables
    x = state(1);     % Position x
    y = state(2);     % Position y
    phi = state(3);   % Yaw angle
    vx = state(4);    % Longitudinal velocity
    vy = state(5);    % Lateral velocity
    omega = state(6); % Yaw rate

    % Slip angles calculation
    alpha_f = - atan((vy + lf * omega) / vx) + delta;
    alpha_r = atan((lr * omega - vy) / vx);

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
