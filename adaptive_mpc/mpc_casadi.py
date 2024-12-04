import casadi as ca
import numpy as np
import time
from parameter_estimation import parameter_estimate
from draw import Draw_MPC_Obstacle, Draw_MPC_point_stabilization_v1

import matplotlib.pyplot as plt

m=2.923
Iz=0.0796
lf=0.168
lr=0.163

T = 0.1                 # time step
N = 45                  # horizon length
vx_max = 10.0             # linear velocity max
vy_max = 7.0             # linear velocity max
omega_max = np.pi   # angular velocity max
u_max = 2.0               # force max of each direction
steering_max = 0.4967

Pf_init = 1  # Initial guess for Pf
Pr_init = 1  # Initial guess for Pr

theta_param = np.array([Pf_init, Pr_init])  # Initial parameter estimates

def car_model(state, u, opt_Pf_Pr):
    small_value = 0.001
    alpha_f = -ca.arctan((state[4] + lf * state[5]) / (state[3] + small_value)) + u[1]
    alpha_r = ca.arctan((-state[4] + lr * state[5]) / (state[3] + small_value))

    # # Tire forces using Simplified Pacejka model
    Ff = opt_Pf_Pr[0] * alpha_f  # Front lateral force
    Fr = opt_Pf_Pr[1] * alpha_r  # Rear lateral force

    Ff_long = u[0]

    dx = state[3] * ca.cos(state[2]) - state[4] * ca.sin(state[2])
    dy = state[3] * ca.sin(state[2]) + state[4] * ca.cos(state[2])
    dphi = state[5]
    dvx = (1 / m) * (u[0] + Ff_long * ca.cos(u[1]) - Ff * ca.sin(u[1]) + m * state[4] * state[5])
    dvy = (1 / m) * (Ff_long * ca.sin(u[1]) + Ff * ca.cos(u[1]) + Fr - m * state[3] * state[5])
    domega = (1 / Iz) * (Ff_long * lf * ca.sin(u[1]) + Ff * lf * ca.cos(u[1]) - Fr * lr)

    return ca.vertcat(dx, dy, dphi, dvx, dvy, domega)


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
    opt_Pf_Pr = opti.parameter(2)

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + car_model(opt_states[i, :], opt_controls[i, :], opt_Pf_Pr).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # weight matrix
    Q = np.diag([5.0, 5.0, 0.0, 1.0, 1.0, 1.0])
    R = np.diag([0.1, 0.1])

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

    opts_setting = {'ipopt.max_iter': 10000,
                        'ipopt.print_level': 0,  # Set higher for detailed output
                        'print_time': False,
                        'ipopt.acceptable_tol': 1e-6,
                        'ipopt.acceptable_obj_change_tol': 1e-6
                    }
    opti.solver('ipopt', opts_setting)

    # The final state
    final_state = np.array([5, 5, 0.0, 0.0, 0.0, 0.0])
    opti.set_value(opt_xs, final_state)

    # The initial state
    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current_state = init_state.copy()
    next_states = np.zeros((N+1, 6))
    u_c = []
    t_c = [t0]  # for the time
    xx = [current_state]
    sim_time = 100.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(current_state - final_state) > 0.15 and mpciter - sim_time/T < 0.0):
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)
        # opti.set_value(opt_Pf_Pr, theta_param)
        opti.set_value(opt_Pf_Pr, [0,0])

        # # # # set optimizing target withe init guess
        # opti.set_initial(opt_controls, u0)# (N, 3)
        # opti.set_initial(opt_states, next_states) # (N+1, 6)
        
        # solve the problem once again
        print('theta_param: ', theta_param)

        try:
            sol = opti.solve()
        except Exception as e:
            print(e)
            print('Exiting the loop as solver failed')
            break

        # obtain the control input
        u_res = sol.value(opt_controls)
        u_c.append(u_res[0, :])
        next_states_pred = sol.value(opt_states)

        theta_param, state_next = parameter_estimate(current_state, u_res[0,:], T, theta_param)
        
        # state_next = next_states_pred[1, :]
        print('state_next: ', state_next)
        
        # # for next loop
        t0 += T
        t_c.append(t0)
        mpciter = mpciter + 1
        xx.append(state_next)
        current_state = state_next
        print('final error {}'.format(np.linalg.norm(final_state[:2]-current_state[:2])))

    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))

    # after loop
    print(mpciter)
    print('final error {}'.format(np.linalg.norm(final_state-current_state)))
    
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3,
                                                init_state=init_state,
                                                target_state=final_state,
                                                robot_states=np.array(xx),
                                                export_fig=True)
    
    # plot velocity values
    fig, ax = plt.subplots()
    ax.plot(t_c, [x[4] for x in xx], label='vx')
    ax.plot(t_c, [x[5] for x in xx], label='vy')

    ax.legend()
    ax.grid()

    fig.savefig('velocity_values.png')

    # plt.show()