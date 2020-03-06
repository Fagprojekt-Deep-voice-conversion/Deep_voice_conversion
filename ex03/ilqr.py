"""
For linesearch implement method described in (TET12) (we will use regular iLQR, not DDP!)
for the non-linesearch method, see (Har20, Alg 1).

References:
  [TET12] Yuval Tassa, Tom Erez, and Emanuel Todorov. Synthesis and stabilization of complex behaviors through online trajectory optimization. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 4906–4913. IEEE, 2012. (See tassa2012.pdf). 
  [Har20] James Harrison. Optimal and learning-based control combined course notes. (See AA203combined.pdf), 2020. 
"""
import warnings
import numpy as np
from irlc.ex03.dlqr import LQR
import matplotlib.pyplot as plt
import time

def ilqr_basic(env, N, x0, us_init=None, n_iterations=500):
    '''
    Basic ilqr. I.e. Algorithm 1 in (Har20). Our notation (x_bar, etc.) will be consistent with the lecture slides
    '''
    mu, alpha = 1, 1 # We will get back to these. For now, just let them have defaults and don't change them
    n, m = env.state_size, env.action_size
    u_bar = [np.random.uniform(-1, 1,(env.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * (N)
    
    """
    Initialize nominal trajectory xs, us using us and x0 (i.e. simulate system from x0 using action sequence us). 
    The simplest way to do this is to call forward_pass with all-zero sequence of control vector/matrix l, L.
    """
    l, L = [np.zeros(m,)] * (N), [np.zeros((m,n))] * (N)

    x_bar, u_bar = forward_pass(env, x_bar, u_bar, L, l)
    # raise NotImplementedError("Initialize x_bar, u_bar here")
    J_hist = []
    for i in range(n_iterations):

        """
        Compute derivatives around trajectory and cost estimate J of trajectory. To do so, use the get_derivatives
        function        
        """
        (f_x, f_u), (L, L_x, L_u, L_xx, L_ux, L_uu) = get_derivatives(env, x_bar, u_bar)
        J = compute_J(env, x_bar, u_bar)
        #raise NotImplementedError("Compute J and derivatives f_x, f_u, ....")
        """  Backward pass: Obtain feedback law matrices l, L using the backward_pass function.
        """
        # TODO: 1 lines missing.
        L, l = backward_pass(f_x, f_u, L_x, L_u, L_xx, L_ux, L_uu)
       # raise NotImplementedError("Compute L, l = .... here")
        """ Forward pass: Given L, l matrices computed above, simulate new (optimal) action sequence using
        (TET12, eq.12). In the lecture slides, this is similar to how we compute u^*_k and x_k
        Once they are computed, iterate the iLQR algorithm by setting x_bar, u_bar equal to these values
        """
        # TODO: 1 lines missing.
        x_bar, u_bar = forward_pass(env, x_bar, u_bar, L, l)
        #raise NotImplementedError("Compute x_bar, u_bar = ...")
        print(f"{i}> J={J:4g}, change in cost since last iteration {0 if i == 0 else J-J_hist[-1]:4g}")
        J_hist.append(J)
    return x_bar, u_bar, J_hist

def ilqr_linesearch(env, N, x0, n_iterations, us_init=None, tol=1e-6):
    """
    For linesearch implement method described in (TET12) (we will use regular iLQR, not DDP!)
    """
    # The range of alpha-values to try out in the linesearch
    # plus parameters relevant for regularization scheduling. See (TET12, eq.12).
    alphas = 1.1 ** (-np.arange(10) ** 2)
    mu_min = 1e-6
    mu_max = 1e10
    Delta_0 = 2
    mu = 1.0
    Delta = Delta_0

    n, m = env.state_size, env.action_size
    u_bar = [np.random.uniform(-1, 1, (env.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * (N)
    # Initialize nominal trajectory xs, us (same as in basic linesearch)
    l, L = [np.zeros(m,)] * (N), [np.zeros((m,n))] * (N)
    x_bar, u_bar = forward_pass(env, x_bar, u_bar, L, l)
    J_hist = []

    converged = False
    # We will implement the three steps in (TET12, Page 4908, top-right) (also check out slides)
    for i in range(n_iterations):
        alpha_was_accepted = False
        """ Step 1: Compute derivatives around trajectory and cost estimate of trajectory.
        (copy-paste from basic implementation). In our implementation, J_bar = J_{u^star}(x_0) """
        # TODO: 2 lines missing.
        (f_x, f_u), (L, L_x, L_u, L_xx, L_ux, L_uu) = get_derivatives(env, x_bar, u_bar)
        J_bar = compute_J(env, x_bar, u_bar)
        #raise NotImplementedError("Obtain derivatives f_x, f_u, ... as well as cost of trajectory J_bar = ...")
        try:
            """
            Step 2: Backward pass to obtain control law (l, L). Same as before so more copy-paste
            """
            L,l = backward_pass(f_x, f_u, L_x, L_u, L_xx, L_ux, L_uu)
            #raise NotImplementedError("Obtain l, L = ... in backward pass")
            """
            Step 3: Forward pass and alpha scheduling.
            Decrease alpha and check condition (TET12, Eq.(13)). Apply the regularization scheduling as needed. """
            for alpha in alphas:
                x_hat, u_hat = forward_pass(env, x_bar, u_bar, L=L, l=l, alpha=alpha) # Simulate trajectory using this alpha
                J_new = compute_J(env, x_hat, u_hat)
                #raise NotImplementedError("Compute J_new = ... as the cost of trajectory x_hat, u_hat")

                if J_new < J_bar:
                    """ Linesearch proposed trajectory accepted! Set current trajectory equal to x_hat, u_hat.
                    We will be a bit lazy and implement (TET12, Eq 13) as the following.
                    I.e. we replace Delta(J) with just J_opt. """
                    if np.abs((J_bar - J_new) / J_bar) < tol:
                        converged = True  # Method does not seem to decrease J; converged. Break and return.

                    J_bar = J_new
                    x_bar, u_bar = x_hat, u_hat
                    '''
                    The update was accepted and you should change the regularization term mu, 
                     and the related scheduling term Delta. The change should be either increase or decrease from
                     (TET12, p. 4908); which one is it? Implement the right choice below                    
                    '''
                    # Decreasing mu
                    Delta = min(1/Delta_0, Delta/Delta_0)
                    mu = mu * Delta if mu * Delta > mu_min else 0
                    #raise NotImplementedError("Delta, mu = ...")
                    alpha_was_accepted = True # accept this alpha
                    break
        except np.linalg.LinAlgError as e:
            # Matrix in dlqr was not positive-definite and this diverged
            warnings.warn(str(e))

        if not alpha_was_accepted:
            ''' No alphas were accepted, which is not too hot. Regularization should change, but how?
            look at  (TET12, p. 4908), select the right option, and update mu/Delta
            '''
            # Increase mu
            Delta = max(Delta_0, Delta * Delta_0)
            mu = max(mu_min, mu * Delta)
            #raise NotImplementedError("Delta, mu = ...")

            if mu_max and mu >= mu_max:
                raise Exception("Exceeded max regularization term; we are stuffed.")

        dJ = 0 if i == 0 else J_bar-J_hist[-1]
        info = "converged" if converged else ("accepted" if alpha_was_accepted else "failed")
        print(f"{i}> J={J_bar:4g}, decrease in cost {dJ:4g} ({info}).\nx[N]={x_bar[-1].round(2)}")
        J_hist.append(J_bar)
        if converged:
            break
    return x_bar, u_bar, J_hist



def backward_pass(f_x, f_u, L_x, L_u, L_xx, L_ux, L_uu, _mu=1):
    """
    Get L,l feedback law given linearization around nominal trajectory (the 'Q'-terms in (Har20, Alg 1)).
    To do so, simply call LQR with appropriate inputs.
    """
    # TODO: 6 lines missing.
    A, B = f_x, f_u
    Q, R = L_xx[:-1], L_uu
    QN, qN = L_xx[-1], L_x[-1]
    H = L_ux
    q, r = L_x[:-1], L_u

    #raise NotImplementedError("")
    (L, l), (V, v, vc) = LQR(A=A, B=B, R=R, Q=Q, QN=QN, H=H, q=q, qN=qN, r=r, mu=_mu)
    return L,l

def compute_J(env, xs, us):
    """
    Helper function which computes the cost of the trajectory. 
    
    Input: 
        xs: States (N+1) x [(state_size)]
        us: Actions N x [(state_size)]
        
    Returns:
        Trajectory's total cost.
    """
    N = len(us)
    JN = env.gN(xs[-1])
    return sum(map(lambda args: env.g(*args), zip(xs[:-1], us, range(N)))) + JN

def get_derivatives(env, x_bar, u_bar):
    """
    Compute derivatives for system dynamics around the given trajectory. should be handled using
    env.f and env.g+env.gN.

    f_x, f_u has the same meaning as in (TET12, Eq. 5) or A_k, B_k in the lecture slides. I.e. these are
    lists of the derivatives of the system dynamics wrt. x and u.

    Meanwhile the terms L, L_x, ... Should be compared to the V-terms in (TET12, Eq.5) or (perhaps simpler)
    to the derivatives of the term c in the lecture slides. i.e.
    >>> L[k] = c_k,
    >>> L_x[k] = c_{x,k}
    >>> L_ux[k] = c_{ux,k}

    and so on. These derivatives will be returned as lists of matrices/vectors as appropriate. Note that in particular
    L will be a N+1 list of the cost terms, such that J = sum(L) is the total cost of the trajectory.
    """
    N = len(u_bar)
    """ Compute f_x, f_u (lists of matrices of length N)
    Recall env.f has output
        x, f_x[i], f_u[i], _, _, _ = env.f(x, u, i, compute_jacobian=True)
    """
    # raise NotImplementedError("")
    """ Compute derivatives of the cost function. For terms not including u these should be of length N+1 
    (because of gN!), for the other lists of length N
    recall env.g has output:
        L[i], L_x[i], L_u[i], L_xx[i], L_ux[i], L_uu[i] = env.g(x, u, i, terminal=False, compute_gradients=True)
    """
    f_x, f_u = [None] * (N), [None] * N
    L, L_x, L_u, L_xx, L_ux, L_uu = [None] * (N+1), [None] * (N+1), [None] * (N), [None] * (N+1), [None] * (N), [None] * (N)
    for i in range(N):
        x, f_x[i], f_u[i], _, _, _ = env.f(x_bar[i], u_bar[i], i, compute_jacobian=True)
        L[i], L_x[i], L_u[i], L_xx[i], L_ux[i], L_uu[i] = env.g(x_bar[i], u_bar[i], i, terminal=False, compute_gradients=True)
    # TODO: 2 lines missing.
    #raise NotImplementedError("")
    # Concatenate the derivatives associated with the last time point N.
    LN, L_xN, L_xxN = env.gN(x_bar[N], compute_gradients=True)
    L[N] = LN
    L_x[N] = L_xN
    L_xx[N] = L_xxN
    return (f_x, f_u), (L, L_x, L_u, L_xx, L_ux, L_uu)

def forward_pass(env, x_bar, u_bar, L, l, alpha=1.0):
    """Applies the controls for a given trajectory.

    Args:
        x_bar: Nominal state path [N+1, state_size].
        u_bar: Nominal control path [N, action_size].
        l: Feedforward gains [N, action_size].
        L: Feedback gains [N, action_size, state_size].
        alpha: Line search coefficient.

    Returns:
        Tuple of
            x: state path [N+1, state_size] simulated by the system
            us: control path [N, action_size] new control path
    """
    N = len(u_bar)
    x = [None] * (N+1)
    u_star = [None] * N
    x[0] = x_bar[0].copy()

    for i in range(N):
        """ Compute using (TET12, Eq (12))
        u_{i} = ...
        """
        u_star[i] = u_bar[i] + alpha * l[i] + L[i] @ (x[i] - x_bar[i])

        """ Compute using (TET12, Eq (8c))
        x_{i+1} = ...
        """
        x[i+1] = env.f(x[i], u_star[i], i)

    return x, u_star
