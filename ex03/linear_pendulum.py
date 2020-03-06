"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import matplotlib.pyplot as plt
import numpy as np
from irlc.ex03.cost import QRCost
from irlc.ex03.dp_cartpole_env import CartpoleSinCosEnvironment
from irlc.ex03.dp_pendulum_env import PendulumSinCosEnvironment
from irlc.ex03.dlqr import LQR
import time
from irlc import savepdf

def run_lqr(env, K, N, x_bar, u_bar, x0, render=False):
    x = x0
    
    xs = [x]
    us = []
    for t in range(N):
        u = u_bar + (K[t] @ (x - x_bar)).squeeze() #!b
        x = env.f_discrete(x, u) #!b
        us.append(u)
        xs.append(x)
        if render:
            time.sleep(0.05)
            env.render(x)
    if render:
        env.viewer.close()
    
    return xs, us

def linear_cartpole(N,dt,render=False):
    # This is not currently used.
    max_force = 10.0
    pole_length =1
    goal_angle = 0
    x_goal = np.array([0, 0, np.sin(goal_angle), np.cos(goal_angle), 0]) #final state: up
    init_angle = 20 * np.pi * 2 / 360
    x_init_state = (0, 0, np.sin(init_angle), np.cos(init_angle), 0)
    Q = np.diag([1.0, 1.0, 100.0, 0.0, 1.0])

    QN = 1000 * Q
    # Instantaneous control cost.
    R = np.array([[0.1]])
    cost = QRCost(Q, R, QN=QN, x_goal=x_goal)
    env = CartpoleSinCosEnvironment(dt=dt, cost=cost, l=pole_length,min_bounds = -max_force, max_bounds =max_force)
    xs, us = linearize(env, N, x_init_state, x_goal, render=render)

def linear_pendulum(N,dt,render=False):
    state_size = PendulumSinCosEnvironment.state_size
    x_goal = PendulumSinCosEnvironment.x_upright
    u_goal = np.asarray([0]) # do not use any force
    """ set up cost matrices """
    Q = np.eye(state_size)
    Q[1, 1] = 0
    QN = 100 * Q
    
    Q = np.diag([ 10.0, 0.0, 10.0])
    R = np.array([[1]])
    """ set up the environment """
    cost = QRCost(Q, R, QN=QN, x_goal=x_goal)

    env = PendulumSinCosEnvironment(dt, cost=cost, max_bounds=5)
    init_angle = 15 * np.pi * 2 / 360
    """
    Initial state. We will linearize around this
    """
    x_init_state = ( np.sin(init_angle), np.cos(init_angle), 0)
    u_init = (0,)

    xs, us = linearize(env, N, x_bar=x_goal, u_bar=u_goal, x0=x_init_state, u0=u_init, render=render)
    t = np.arange(N+1) * dt
    plt.subplot(2,2,1)
    plt.plot(xs[:,0], xs[:,1], 'k-')
    plt.title('x, y')
    plt.subplot(2,2,2)
    plt.plot(t, xs[:,0], 'k-')
    plt.plot(t, xs[:,1], 'r-')
    plt.title('x (black) y (red)')
    plt.subplot(2, 2, 3)
    plt.plot(t[:-1], us)
    plt.title('u')
    savepdf("linear_pendulum")
    plt.show()

def linearize(env, N, x0, u0, x_bar, u_bar, render = False):
    n = env.state_size
    f_z = env.f_z(x0, u0)
    """
    Use f_z to define A, B. The cost-matrices can be obtained from env.cost. What happens with the linear terms?
    (hint: the expansion in the dynamics is around x_init_state). 
    When done, collect the matrices you need to call LQR with and run it. 
    
    LQR needs a total of 5 inputs total.        
    """

    
    A = f_z[:,0:n]
    B = f_z[:,n].reshape(n,-1)
    Q = env.cost.Q
    R = env.cost.R
    (L,l), (V,v,vc) = LQR(A=[A]*N, B=[B]*N, d=None, Q=[Q]*N, R=[R]*N, QN=env.cost.Q_terminal)
    # raise NotImplementedError("")
    
    xs, us = run_lqr(env, L, N, x_bar=x_bar, u_bar=u_bar, x0=x0, render=render)
    xs, us = np.asarray(xs), np.asarray(us)
    return xs, us

if __name__ == "__main__":
    render = False
    N,dt = 200, 0.05
    linear_pendulum(N, dt, render)
    # linear_cartpole(N,dt,render)
