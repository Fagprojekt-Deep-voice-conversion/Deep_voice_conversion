"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import matplotlib.pyplot as plt
import numpy as np
from irlc.ex03.cost import QRCost
from irlc.ex03.dp_cartpole_env import CartpoleSinCosEnvironment
import time
from irlc.ex03.ilqr_rendovouz_basic import ilqr
from irlc import savepdf

def cartpole(use_linesearch):
    dt = 0.05
    pole_length = 1.0
    x_goal = np.array([0.0, 0.0, np.sin(0), np.cos(0), 0.0])

    # Instantaneous state cost.
    state_size = 5
    Q = np.eye(state_size)
    Q[0, 0] = 1.0
    Q[1, 1] = Q[4, 4] = 0.0
    Q[0, 2] = Q[2, 0] = pole_length
    Q[2, 2] = Q[3, 3] = pole_length**2
    Q = np.diag([0.0, 1.0, 1.0, 0.0, 0.0])
    R = np.array([[0.1]])
    # Terminal state cost.
    Q_terminal = 1000 * np.eye(state_size)

    # Instantaneous control cost.
    cost = QRCost(Q, R, QN=Q_terminal, x_goal=x_goal)

    env = CartpoleSinCosEnvironment(dt=dt, cost=cost, l=pole_length)

    N = 300
    init_angle = 180 * np.pi * 2 / 360
    x0 = np.array([0.0, 0.0, np.sin(init_angle), np.cos(init_angle), 0.0])
    xs, us, J_hist = ilqr(env, N, x0, n_iter=300, use_linesearch=use_linesearch)
    xs0 = xs.copy()
    us = env.transform_actions(us)
    xs = env.reduce_state(xs)
    t = np.arange(N + 1) * dt
    x = xs[:, 0]
    theta = np.unwrap(xs[:, 2])  # Makes for smoother plots.
    theta_dot = xs[:, 3]
    pdf_ex = '_linesearch' if use_linesearch else ''
    ev = 'cartpole_'

    plt.plot(theta, theta_dot)
    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title("Orientation Phase Plot")
    plt.grid()
    savepdf(f"{ev}theta{pdf_ex}")
    plt.show()

    _ = plt.plot(t[:-1], us)
    _ = plt.xlabel("time (s)")
    _ = plt.ylabel("Force (N)")
    _ = plt.title("Action path")
    plt.grid()
    savepdf(f"{ev}action{pdf_ex}")
    plt.show()

    _ = plt.plot(t, x)
    _ = plt.xlabel("time (s)")
    _ = plt.ylabel("Position (m)")
    _ = plt.title("Cart position")
    plt.grid()
    savepdf(f"{ev}position{pdf_ex}")
    plt.show()

    _ = plt.plot(J_hist)
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("Total cost")
    _ = plt.title("Total cost-to-go")
    plt.grid()
    savepdf(f"{ev}J{pdf_ex}")
    plt.show()

    render = True
    if render:
        for i in range(2):
            render_(xs0, env)
            time.sleep(1)
        env.viewer.close()

def render_(xs, env):
    for i in range(xs.shape[0]):
        x = xs[i]
        env.render(x=x)

if __name__ == "__main__":
    cartpole(use_linesearch=True)
