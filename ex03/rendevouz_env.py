"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from irlc.ex03.dp_symbolic_env import DPSymbolicEnvironment
import sympy as sym
from collections import OrderedDict

class RendevouzEnvironment(DPSymbolicEnvironment):
    action_size = 4
    state_size = 8

    def __init__(self,
                 dt,
                 cost=None,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 m=1.0,
                 alpha=0.1,
                 **kwargs):
        """Constructs a Rendevouz  model.
        """

        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        self.sym_m, self.sym_alpha = sym.symbols('m, alpha')
        par_map = OrderedDict()
        par_map[self.sym_m] = m
        par_map[self.sym_alpha] = alpha
        self.par_map = par_map
        super(RendevouzEnvironment, self).__init__(dt=dt, cost=cost)


    def sym_f_discrete(self, xs, us, dt):
        par_map = self.par_map
        m = self.sym_m
        alpha = self.sym_alpha

        xs1 = xs[:4]
        xs2 = xs[4:]

        xp1 = [x1 + x2 * dt for x1, x2 in zip(xs1, xs2)]
        def acceleration(x_dot, u):
            x_dot_dot = x_dot * (1 - alpha * dt / m) + u * dt / m
            return x_dot_dot

        xp2 = [x2 + acceleration(x2, u)*dt for x1, x2, u in zip(xs1, xs2, us)]

        xp = xp1 + xp2
        f = [ff.subs(par_map) for ff in xp]
        return f
