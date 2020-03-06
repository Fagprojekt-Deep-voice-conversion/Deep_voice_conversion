"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from collections import OrderedDict
import sympy as sym
import numpy as np
from gym.envs.classic_control import rendering
from irlc.ex03.dp_symbolic_env import DPSymbolicEnvironment

def dim1(x):
    return np.reshape(x, (x.size,) ) if x is not None else x

class CartpoleSinCosEnvironment(DPSymbolicEnvironment):
    """ Symbolic version of the discrete Cartpole environment. """
    action_size = 1
    state_size = 5

    def __init__(self,
                 dt,
                 cost=None,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 mc=1.0,
                 mp=0.1,
                 l=1.0,
                 g=9.80665,
                 **kwargs):
        """Cartpole dynamics.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N].
            max_bounds: Maximum bounds for action [N].
            mc: Cart mass [kg].
            mp: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                AutoDiffDynamics constructor.

        Note:
            state: [x, x', sin(theta), cos(theta), theta']
            action: [F]
            theta: 0 is pointing up and increasing clockwise.
        """

        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        self.l = l
        self.m_p, self.m_c, self.sym_l, self.sym_g = sym.symbols('m_p, m_c, l, g')

        par_map = OrderedDict()
        par_map[self.m_p] = mp
        par_map[self.m_c] = mc
        par_map[self.sym_l] = l
        par_map[self.sym_g] = g
        self.par_map = par_map

        super(CartpoleSinCosEnvironment, self).__init__(dt=dt, cost=cost)

    def render(self, x=None):
        # the render function we use assumes parameterization in terms of these.
        sin_theta = x[2]
        cos_theta = x[3]
        theta = np.arctan2(sin_theta, cos_theta)
        x_theta = [x[0], x[1], theta, x[4]]

        render_cartpole(self, x=x_theta, mode="human")

    def x_cont2x_discrete(self, xs):
        """
        converts state space with theta to state with sin cos
        :param xs: x with theta in space
        :return:  x with sin and cos
        """
        return [xs[0], xs[1], sym.sin(xs[2]), sym.cos(xs[2]), xs[3]]

    def x_discrete2x_cont(self, xs):
        return [xs[0], xs[1], sym.atan2(xs[2], xs[3]), xs[4]]

    def sym_f_discrete(self, xs, us, dt):
        """

        :param xs: current state
        :param us: force
        :param dt: discrete time length
        :return: next state
        """
        min_bounds = self.min_bounds
        max_bounds = self.max_bounds
        par_map = self.par_map
        mp = self.m_p
        mc = self.m_c
        l = self.sym_l
        g = self.sym_g
        x_ = xs[0]
        x_dot = xs[1]
        sin_theta = xs[2]
        cos_theta = xs[3]
        theta_dot = xs[4]

        F = sym.tanh(us[0]) * (max_bounds - min_bounds) / 2.0
        # Define dynamics model as per Razvan V. Florian's
        # "Correct equations for the dynamics of the cart-pole system".
        # Friction is neglected.

        # Eq. (23)
        temp = (F + mp * l * theta_dot ** 2 * sin_theta) / (mc + mp)
        numerator = g * sin_theta - cos_theta * temp
        denominator = l * (4.0 / 3.0 - mp * cos_theta ** 2 / (mc + mp))
        theta_dot_dot = numerator / denominator

        # Eq. (24)
        x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)

        # Deaugment state for dynamics.
        theta = sym.atan2(sin_theta, cos_theta)
        next_theta = theta + dt * theta_dot

        xp = [
            x_ + x_dot * dt,
            x_dot + x_dot_dot * dt,
            sym.sin(next_theta),
            sym.cos(next_theta),
            theta_dot * .99 + theta_dot_dot * dt,
        ]

        f = [ff.subs(par_map) for ff in xp]  # insert the actual values for length, g, etc. into the equation
        return f

    def transform_actions(self, us):
        return np.tanh(us) * (self.max_bounds - self.min_bounds) / 2

    def reduce_state(self, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:

            [x, x', sin(theta), cos(theta), theta'] -> [x, x', theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            x, x_dot, sin_theta, cos_theta, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 1].reshape(-1, 1)
            sin_theta = state[..., 2].reshape(-1, 1)
            cos_theta = state[..., 3].reshape(-1, 1)
            theta_dot = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])

def render_cartpole(env, x=None, mode='human', close=False):
    if not hasattr(env, 'viewer'):
        env.viewer = None

    if x is None:
        x = env.state
    if close:
        if env.viewer is not None:
            env.viewer.close()
            env.viewer = None
        return None

    screen_width = 600
    screen_height = 400

    world_width = 8  # max visible position of cart
    scale = screen_width / world_width
    carty = 200  # TOP OF CART
    polewidth = 8.0
    # return
    polelen = scale * env.l # 0.6 or self.l

    cartwidth = 40.0
    cartheight = 20.0

    if env.viewer is None:
        env.viewer = rendering.Viewer(screen_width, screen_height)

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        env.carttrans = rendering.Transform()
        cart.add_attr(env.carttrans)
        cart.set_color(1, 0, 0)
        env.viewer.add_geom(cart)

        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(0, 0, 1)
        env.poletrans = rendering.Transform(translation=(0, 0))
        pole.add_attr(env.poletrans)
        pole.add_attr(env.carttrans)
        env.viewer.add_geom(pole)

        env.axle = rendering.make_circle(polewidth / 2)
        env.axle.add_attr(env.poletrans)
        env.axle.add_attr(env.carttrans)
        env.axle.set_color(0.1, 1, 1)
        env.viewer.add_geom(env.axle)

        # Make another circle on the top of the pole
        env.pole_bob = rendering.make_circle(polewidth / 2)
        env.pole_bob_trans = rendering.Transform()
        env.pole_bob.add_attr(env.pole_bob_trans)
        env.pole_bob.add_attr(env.poletrans)
        env.pole_bob.add_attr(env.carttrans)
        env.pole_bob.set_color(0, 0, 0)
        env.viewer.add_geom(env.pole_bob)

        env.wheel_l = rendering.make_circle(cartheight / 4)
        env.wheel_r = rendering.make_circle(cartheight / 4)
        env.wheeltrans_l = rendering.Transform(translation=(-cartwidth / 2, -cartheight / 2))
        env.wheeltrans_r = rendering.Transform(translation=(cartwidth / 2, -cartheight / 2))
        env.wheel_l.add_attr(env.wheeltrans_l)
        env.wheel_l.add_attr(env.carttrans)
        env.wheel_r.add_attr(env.wheeltrans_r)
        env.wheel_r.add_attr(env.carttrans)
        env.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
        env.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
        env.viewer.add_geom(env.wheel_l)
        env.viewer.add_geom(env.wheel_r)

        env.track = rendering.Line((0, carty - cartheight / 2 - cartheight / 4),
                                   (screen_width, carty - cartheight / 2 - cartheight / 4))
        env.track.set_color(0, 0, 0)
        env.viewer.add_geom(env.track)

    if x is None:
        return None

    cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    env.carttrans.set_translation(cartx, carty)
    env.poletrans.set_rotation(-x[2] + 0 * np.pi)
    env.pole_bob_trans.set_translation(-env.l * np.sin(x[2]), env.l * np.cos(x[2]))

    return env.viewer.render(return_rgb_array=mode == 'rgb_array')
