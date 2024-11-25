import casadi as ca
import numpy as np
import time
from parameter_estimation import parameter_estimate
from draw import Draw_MPC_Obstacle, Draw_MPC_point_stabilization_v1

m=1500
Iz= 3000
lf=1.1
lr=1.6
Pf=6000
Pr=4000

T = 0.1                 # time step
N = 30                  # horizon length
vx_max = 400.0             # linear velocity max
vy_max = 100.0             # linear velocity max
omega_max = np.pi   # angular velocity max
u_max = 100               # force max of each direction
steering_max = np.pi/6

Pf_init = 100  # Initial guess for Pf
Pr_init = 500  # Initial guess for Pr

theta_param = np.array([Pf_init, Pr_init])  # Initial parameter estimates
P = np.eye(2) * 5  # Example covariance matrix

obs_x = [5.5]
obs_y = [5.0]
obs_diam = 0.5
bias = 0.02

def shift(T, t0, x0, u, x_n, f):
    f_value = f(x0, u[0])
    st = x0 + f_value*T
    t = t0 + T
    zeros_u = np.zeros(u[-1:].shape)
    u_end = np.concatenate((u[1:], zeros_u))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return t, st, u_end, x_n

def car_model(state, u):
    alpha_f = ca.if_else(state[3] <= 1, 0, -ca.arctan((state[4] + lf * state[5]) / (state[3])) + u[1])
    alpha_r = ca.if_else(state[3] <= 1, 0, ca.arctan((-state[4] + lr * state[5]) / (state[3])))

    Ff = Pf * alpha_f
    Fr = Pr * alpha_r

    Ff_long = u[0]*ca.cos(u[1])

    dx = state[3] * ca.cos(state[2]) - state[4] * ca.sin(state[2])
    dy = state[3] * ca.sin(state[2]) + state[4] * ca.cos(state[2])
    dphi = state[5]
    dvx = (1 / m) * (u[0] + Ff_long * ca.cos(u[1]) -Ff * ca.sin(u[1]) + m * state[4] * state[5])
    dvy = (1 / m) * (Ff_long * ca.sin(u[1]) + Ff * ca.cos(u[1]) + Fr - m * state[4] * state[5])
    domega = (1 / Iz) * (Ff_long * lf * ca.sin(u[1]) + Ff * lf * ca.cos(u[1]) - Fr * lr)

    return ca.vertcat(dx, dy, dphi, dvx, dvy, domega)


def car_model_np(state, u):

    u[0] = ca.if_else(ca.logic_and(u[0] < 0, state[3] < 0.01), 0, u[0])
    # Slip angles calculation
    if(state[3] <= 1):
        alpha_f = 0
        alpha_r = 0
    else:
        alpha_f = -ca.arctan((state[4] + lf * state[5]) / (state[3])) + u[1]
        alpha_r = ca.arctan((-state[4] + lr * state[5]) / (state[3]))
    # Tire forces using Simplified Pacejka model
    Ff = Pf * alpha_f  # Front lateral force
    Fr = Pr * alpha_r  # Rear lateral force

    Ff_long = u[0]*ca.cos(u[1])

    dx = state[3] * ca.cos(state[2]) - state[4] * ca.sin(state[2])
    dy = state[3] * ca.sin(state[2]) + state[4] * ca.cos(state[2])
    dphi = state[5]
    dvx = (1 / m) * (u[0] + Ff_long * ca.cos(u[1]) -Ff * ca.sin(u[1]) + m * state[4] * state[5])
    dvy = (1 / m) * (Ff_long * ca.sin(u[1]) + Ff * ca.cos(u[1]) + Fr - m * state[4] * state[5])
    domega = (1 / Iz) * (Ff_long * lf * ca.sin(u[1]) + Ff * lf * ca.cos(u[1]) - Fr * lr)
    # Return state derivatives
    dstate = np.array([dx, dy, dphi, dvx, dvy, domega])
    return dstate

if __name__ == "__main__":
    opti = ca.Opti()
    # control variables, Steering angle and Thrust
    opt_controls = opti.variable(N, 2)
    u1 = opt_controls[:, 0]
    u2 = opt_controls[:, 1]
    
    # state variable: position and velocity
    opt_states = opti.variable(N+1, 6)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    vx = opt_states[:, 3]
    vy = opt_states[:, 4]
    omega = opt_states[:, 5]

    # parameters
    opt_x0 = opti.parameter(6)
    opt_xs = opti.parameter(6)

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + car_model(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # weight matrix
    Q = np.diag([5.0, 5.0, 10.0, 0.5, 0.5, 0.5])
    R = np.diag([0.01, 0.01])

    # cost function
    obj = 0
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :] - opt_xs.T), Q, (opt_states[i, :] - opt_xs.T).T]) \
                     + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
    opti.minimize(obj)

    # boundary and control conditions
    opti.subject_to(opti.bounded(-100.0, x, 100.0))
    opti.subject_to(opti.bounded(-100.0, y, 100.0))
    opti.subject_to(opti.bounded(-vx_max, vx, vx_max))
    opti.subject_to(opti.bounded(-vy_max, vy, vy_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))
    opti.subject_to(opti.bounded(-u_max, u1, u_max))
    opti.subject_to(opti.bounded(-steering_max, u2, steering_max))

    opts_setting = {'ipopt.max_iter': 1000,
                        'ipopt.print_level': 0,  # Set higher for detailed output
                        'print_time': False,
                        'ipopt.acceptable_tol': 1e-6,
                        'ipopt.acceptable_obj_change_tol': 1e-6
                    }
    opti.solver('ipopt', opts_setting)

    # The final state
    final_state = np.array([3, 3, 0.0, 0.0, 0.0, 0.0])
    opti.set_value(opt_xs, final_state)

    # The initial state
    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u0 = np.zeros((N, 2))
    current_state = init_state.copy()
    next_states = np.zeros((N+1, 6))
    x_c = []    # contains for the history of the state
    u_c = []
    t_c = [t0]  # for the time
    xx = []
    sim_time = 200.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(current_state - final_state) > 1e-2 and mpciter - sim_time/T < 0.0):
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)

        # # set optimizing target withe init guess
        opti.set_initial(opt_controls, u0)# (N, 3)
        opti.set_initial(opt_states, next_states) # (N+1, 6)
        
        # solve the problem once again
        sol = opti.solve()
        theta_param = parameter_estimate(current_state, u0[0,:], T, theta_param)
        Pf=theta_param[0]
        Pr=theta_param[1]

        # obtain the control input
        u_res = sol.value(opt_controls)
        u_c.append(u_res[0, :])
        t_c.append(t0)
        next_states_pred = sol.value(opt_states)# prediction_state(x0=current_state, u=u_res, N=N, T=T)
        
        # next_states_pred = prediction_state(x0=current_state, u=u_res, N=N, T=T)
        x_c.append(next_states_pred)
        
        # for next loop
        t0, current_state, u0, next_states = shift(T, t0, current_state, u_res, next_states_pred, car_model_np)
        mpciter = mpciter + 1
        xx.append(current_state)
        print('final error {}'.format(np.linalg.norm(final_state-current_state)))

    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))

    # after loop
    print(mpciter)
    print('final error {}'.format(np.linalg.norm(final_state-current_state)))

    # draw_result = Draw_MPC_Obstacle(rob_diam=rob_diam,
    #                                 init_state=init_state,
    #                                 target_state=final_state,
    #                                 robot_states=np.array(xx),
    #                                 obstacle=[obs_x, obs_y, obs_diam/2.])
    
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3,
                                                init_state=init_state,
                                                target_state=final_state,
                                                robot_states=np.array(xx))