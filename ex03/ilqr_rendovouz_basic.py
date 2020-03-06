"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import numpy as np
from irlc.ex03.rendevouz_env import RendevouzEnvironment
from irlc.ex03.cost import QRCost
import matplotlib.pyplot as plt
from irlc.ex03.ilqr import ilqr_basic, ilqr_linesearch
from irlc import savepdf

def ilqr(env, N, x0, n_iter, use_linesearch):
    if not use_linesearch:
        xs, us, J_hist = ilqr_basic(env, N, x0, n_iterations=n_iter) 
    else:
        xs, us, J_hist = ilqr_linesearch(env, N, x0, n_iterations=n_iter, tol=1e-6)
    xs, us = np.stack(xs), np.stack(us)
    return xs, us, J_hist

def solve_rendovouz(use_linesearch=False):
    dt = 0.1  # time discretization
    state_size = RendevouzEnvironment.state_size
    action_size = RendevouzEnvironment.action_size
    Q = np.eye(state_size)
    Q[0, 2] = Q[2, 0] = -1
    Q[1, 3] = Q[3, 1] = -1
    R = 0.1 * np.eye(action_size)
    cost = QRCost(Q, R)
    env = RendevouzEnvironment(dt=dt, cost=cost)

    N = 200  # Number of time steps in trajectory.
    x0 = np.array([0, 0, 10, 10, 0, -5, 5, 0])  # Initial state.

    xs, us, J_hist = ilqr(env, N, x0, n_iter=10, use_linesearch=use_linesearch)

    x_0 = xs[:, 0]
    y_0 = xs[:, 1]
    x_1 = xs[:, 2]
    y_1 = xs[:, 3]
    x_0_dot = xs[:, 4]
    y_0_dot = xs[:, 5]
    x_1_dot = xs[:, 6]
    y_1_dot = xs[:, 7]

    pdf_ex = '_linesearch' if use_linesearch else ''
    ev = 'rendevouz_'
    _ = plt.title("Trajectory of the two omnidirectional vehicles")
    _ = plt.plot(x_0, y_0, "r")
    _ = plt.plot(x_1, y_1, "b")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}trajectory{pdf_ex}')
    plt.show()

    t = np.arange(N + 1) * dt
    _ = plt.plot(t, x_0, "r")
    _ = plt.plot(t, x_1, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("x (m)")
    _ = plt.title("X positional paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_x_pos{pdf_ex}')
    plt.show()

    _ = plt.plot(t, y_0, "r")
    _ = plt.plot(t, y_1, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("y (m)")
    _ = plt.title("Y positional paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_y_pos{pdf_ex}')
    plt.show()

    _ = plt.plot(t, x_0_dot, "r")
    _ = plt.plot(t, x_1_dot, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("x_dot (m)")
    _ = plt.title("X velocity paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_vx{pdf_ex}')
    plt.show()

    _ = plt.plot(t, y_0_dot, "r")
    _ = plt.plot(t, y_1_dot, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("y_dot (m)")
    _ = plt.title("Y velocity paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_vy{pdf_ex}')
    plt.show()

    _ = plt.plot(J_hist)
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("Total cost")
    _ = plt.title("Total cost-to-go")
    savepdf(f'{ev}cost_to_go{pdf_ex}')
    plt.show()


if __name__ == "__main__":
    solve_rendovouz(use_linesearch=False)

