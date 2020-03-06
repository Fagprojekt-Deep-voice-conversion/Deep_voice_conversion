"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from collections import OrderedDict
import numpy as np
import sympy as sym
from irlc.ex03.dp_symbolic_env import DPSymbolicEnvironment
from irlc.ex03.dp_cartpole_env import render_cartpole

class PendulumSinCosEnvironment(DPSymbolicEnvironment):
    action_size = 1
    state_size = 3
    x_upright = np.array([np.sin(0), np.cos(0), 0.0])
    def __init__(self,
                 dt,
                 cost=None,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 m=1.0,
                 l=1.0,
                 g=9.80665,
                 **kwargs):
        """Constructs an InvertedPendulumDynamics model.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N m].
            max_bounds: Maximum bounds for action [N m].
            m: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                BatchAutoDiffDynamics constructor.

        Note:
            state: [sin(theta), cos(theta), theta']
            action: [torque]
            theta: 0 is pointing up and increasing counter-clockwise.
        """
        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.l = l

        self.sym_m, self.sym_l, self.sym_g = sym.symbols('m, l, g')
        par_map = OrderedDict()
        par_map[self.sym_m] = m
        par_map[self.sym_l] = l
        par_map[self.sym_g] = g
        self.par_map = par_map

        super(PendulumSinCosEnvironment, self).__init__(dt=dt, cost=cost)

    def render(self, x=None):
        # the render function we use assumes parameterization in terms of these.
        sin_theta = x[0]
        cos_theta = x[1]
        theta = np.arctan2(sin_theta, cos_theta)
        x_theta = [0, 0, theta, x[2]]

        render_cartpole(self, x=x_theta, mode="human")

    def sym_f_discrete(self, xs, us, dt): 
        par_map = self.par_map
        m = self.sym_m
        l = self.sym_l
        g = self.sym_g

        sin_theta = xs[0] 
        cos_theta = xs[1]
        theta_dot = xs[2]

        torque = sym.tanh(us[0]) * (self.max_bounds - self.min_bounds) / 2
        theta = sym.atan2(sin_theta, cos_theta)  # Obtain angle theta from sin(theta),cos(theta)
        # Define acceleration.
        theta_dot_dot = g / l * sym.sin(theta) + torque / (m * l ** 2)
        next_theta = theta + theta_dot * dt
        xp = [sym.sin(next_theta),
              sym.cos(next_theta),
              theta_dot + theta_dot_dot * dt] 
        # Note euler integration is instable.
        # Consider multiplying theta_dot with 0.09 above.
        f = [ff.subs(par_map) for ff in xp]
        return f

    def transform_actions(self, us):
        return np.tanh(us) * (self.max_bounds - self.min_bounds) / 2

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:''

            [sin(theta), cos(theta), theta'] -> [theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            sin_theta, cos_theta, theta_dot = state
        else:
            sin_theta = state[..., 0].reshape(-1, 1)
            cos_theta = state[..., 1].reshape(-1, 1)
            theta_dot = state[..., 2].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([theta, theta_dot])
