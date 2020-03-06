"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import numpy as np
from irlc.ex03.dp_pendulum_env import PendulumSinCosEnvironment
from irlc.ex03.dp_cartpole_env import render_cartpole
from irlc.ex03.cost import QRCost
import matplotlib.pyplot as plt
import time
from irlc.ex03.ilqr_rendovouz_basic import ilqr
from irlc import savepdf


def pendulum(use_linesearch):
    dt = 0.02
    pendulum_length = 1.0
    state_size = PendulumSinCosEnvironment.state_size
    # Note that the augmented state is not all 0.
    x_goal = PendulumSinCosEnvironment.x_upright #np.array([np.sin(0), np.cos(0), 0.0])
    Q = np.eye(state_size)
    Q[0, 1] = Q[1, 0] = pendulum_length
    Q[0, 0] = Q[1, 1] = pendulum_length ** 2
    Q[2, 2] = 0.0
    Q_terminal = 1000 * np.eye(state_size)
    R = np.array([[0.1]])*10
    cost = QRCost(Q, R, QN=Q_terminal, x_goal=x_goal)
    bnd = 6
    env = PendulumSinCosEnvironment(dt, cost=cost, l=pendulum_length, min_bounds=-bnd, max_bounds=bnd)
    N = 250
    x0 = np.array([np.sin(np.pi), np.cos(np.pi), 0.0])
    # xs, us, J_hist = ilqr(env, N, x0, n_iter=500, use_linesearch=use_linesearch)
    xs, us, J_hist = ilqr(env, N, x0, n_iter=200, use_linesearch=use_linesearch)

    us = env.transform_actions(us)
    xs = env.reduce_state(xs)

    t = np.arange(N) * dt
    theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
    theta_dot = xs[:, 1]

    pdf_ex = '_linesearch' if use_linesearch else ''
    ev = 'pendulum_'
    _ = plt.plot(theta, theta_dot)
    _ = plt.xlabel("theta (rad)")
    _ = plt.ylabel("theta_dot (rad/s)")
    _ = plt.title("Phase Plot")
    plt.grid()
    savepdf(f"{ev}theta{pdf_ex}")
    plt.show()

    _ = plt.plot(t, us)
    _ = plt.xlabel("time (s)")
    _ = plt.ylabel("Force (N)")
    _ = plt.title("Action path")
    plt.grid()
    savepdf(f"{ev}action{pdf_ex}")
    plt.show()

    _ = plt.plot(J_hist)
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("Total cost")
    _ = plt.title("Total cost-to-go")
    plt.grid()
    savepdf(f"{ev}J{pdf_ex}")
    plt.show()

    render = True #!s=render #!s
    if render:
        for i in range(2):
            render_(xs, env)
            time.sleep(4)

def render_(xs, env):
    for i in range(xs.shape[0]):
        x = [0,0, xs[i][0], xs[i][1]]
        render_cartpole(env, x=x)

if __name__ == "__main__":
    pendulum(use_linesearch=False)
    pendulum(use_linesearch=True)
