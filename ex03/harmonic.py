""" this script examines the relative accuracy of Euler integration """

import matplotlib.pyplot as plt
import numpy as np
from irlc import savepdf

def f_harmonic(x, u, k, m): 
    """
    Helper function. Re-write the problem as

    dx/dt = f(x,u)

    should return a list
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")

def euler_harmonic(x0, u, g, T, N, k=0.1, m=1):
    """
    Simulate a Harmonic oscillator governed by equations:

    d^2 x1 / dt^2 = -k/m x1 + u(x1, t)

    where x1 is the position and u is our externally applied force (the control)
    k is the spring constant and m is the mass. See:

    https://en.wikipedia.org/wiki/Simple_harmonic_motion#Dynamics

    for more details.
    In the code, we will re-write the equations as:

    dx/dt = f(x, u),   u = u_fun(x, t)

    where x = [x1,x2] is now a vector and f is a function of x and the current control.
    here, x1 is the position (same as x in the first equation) and x2 is the velocity.

    The function should return ts, xs, C

    where ts is the N time points t_0, ..., t_{N-1}, xs is a corresponding list [ ..., [x_1(t_k),x_2(t_k)], ...] and C is the cost.
    """
    tt = np.linspace(0, T, N)
    dt = tt[1]-tt[0]
    s = [x0]
    for i, t in enumerate(tt):
        x = s[i]
        sp = f_harmonic(x, u(x, t), k, m)
        s.append( [x[j] + dt * sp[j] for j in range(2)] )
    s = s[:N]
    C = None
    if g is not None:
        C = sum( [ dt*g(s_, 0) for s_ in s[:-1]] )
    return tt, s, C


def u0(x, t):
    return 0

def u_min(x,t): 
    """
    Implement a minimum-seeking  control policy (see problem sheet)
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Implement function body")

def g(x, u):
    """
    Instantaneous potential energy in spring.
    """
    return 1/2 * k * x[0] ** 2

k, m, T = 0.1, 2, 80
x0 = [0, 1]
if __name__ == "__main__":
    N = 100
    tk0 = lambda x: [ x for [x, _] in x]
    tt, x, _ = euler_harmonic(x0=[0, 1], u=u0, g=None, T=T, N=N, k=k, m=m)
    x = tk0(x)
    plt.subplot(1, 2, 1)
    plt.plot(tt, x, label="Euler integration")
    omega = np.sqrt(k/m)
    c1 = 0
    c2 = 1/omega
    x_true = c1 * np.cos(omega * tt) + c2 * np.sin(omega * tt)
    plt.plot(tt, x_true, 'r-', label="True solution")
    plt.xlabel('Seconds'), plt.ylabel('x-position')
    plt.title(f"Simulation using N={N}")
    plt.legend()
    """
    Same as above but with N=1200
    """
    N = 1200
    tt, x, _ = euler_harmonic(x0=[0, 1], u=u0, g=None, T=T, N=N, k=k, m=m)
    x = tk0(x)
    plt.subplot(1, 2, 2)
    plt.plot(tt, x, label="Euler integration")
    x_true = c1 * np.cos(omega * tt) + c2 * np.sin(omega * tt)
    plt.plot(tt, x_true, 'r-', label="True solution")
    plt.xlabel('Seconds'), plt.ylabel('x-position')
    plt.title(f"Simulation using N={N}")
    savepdf("harmonicA.pdf")
    plt.show()

    tt, x, C = euler_harmonic(x0=[0, 1], u=u_min, g=g, T=T, N=N, k=k, m=m)
    x = tk0(x)
    plt.plot(tt, x, label="Euler integration")
    plt.title(f"Simulation using N={N}, trying to find x=0. Cost is C={C:.3f}")
    savepdf("harmonicB.pdf")
    plt.show()
