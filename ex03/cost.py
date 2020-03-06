"""
Quadratic cost functions
"""
import numpy as np
import warnings

class Cost:
    """
    Instantaneous Cost. At this point this class only acts as an interface since I only have one implementation

    NOTE: The terminal cost is a function of x, whereas
          the non-terminal cost can be a function of x, u and k.
    """

    def g(self, x, u, i=None, terminal=False):
        """
        Compute instantaneous cost function and derivatives up to second order

        Returns:
            Instantaneous cost (scalar) and higher order derivatives
        """
        raise NotImplementedError

    def gN(self, x):
        """
        Terminal cost terms + derivatives.
        """
        raise NotImplementedError()

class QRCost(Cost):
    """
    Quadratic Regulator Instantaneous Cost.
    """
    def __init__(self, Q, R, QN=None, x_goal=None, u_goal=None):
        """Constructs a QRCost.
        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            QN: Terminal quadratic state cost matrix
                [state_size, state_size].
            x_goal: Goal state [state_size].
            u_goal: Goal control [action_size].
        """
        self.Q = np.array(Q)
        self.R = np.array(R)

        if QN is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(QN)

        if x_goal is None:
            self.x_goal = np.zeros(Q.shape[0])
        else:
            self.x_goal = np.array(x_goal)

        if u_goal is None:
            self.u_goal = np.zeros(R.shape[0])
        else:
            self.u_goal = np.array(u_goal)

        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        super(QRCost, self).__init__()

    def gN(self, x):
        """Instantaneous cost function and gradiants
        Args:
            x: Current state [state_size].
        Returns:
            Instantaneous cost (scalar).

        """
        Q = self.Q_terminal
        x_diff = x - self.x_goal
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)
        L = squared_x_cost
        Q_plus_Q_T = self._Q_plus_Q_T_terminal
        c_x = x_diff.T.dot(Q_plus_Q_T)
        c_xx = self._Q_plus_Q_T_terminal
        return L, c_x, c_xx

    def g(self, x, u, i=None, terminal=False):
        """Instantaneous cost function and gradiants
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_goal
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            warnings.warn("use gN instead when calling cost function")
            c = squared_x_cost
            c_u = np.zeros_like(self.u_goal)
            c_uu = np.zeros_like(self.R)
        else:
            u_diff = u - self.u_goal
            c = squared_x_cost + u_diff.T.dot(R).dot(u_diff)
            c_u = u_diff.T.dot(self._R_plus_R_T)
            c_uu = self._R_plus_R_T

        # wrt x
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        c_x = x_diff.T.dot(Q_plus_Q_T)

        # wrt xx
        c_xx = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        c_ux = np.zeros((self.R.shape[0], self.Q.shape[0]))
        return c, c_x, c_u, c_xx, c_ux, c_uu
