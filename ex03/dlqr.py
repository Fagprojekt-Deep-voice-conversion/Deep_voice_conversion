"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.

References:
  [Har20] James Harrison. Optimal and learning-based control combined course notes. (See AA203combined.pdf), 2020. 
  [TET12] Yuval Tassa, Tom Erez, and Emanuel Todorov. Synthesis and stabilization of complex behaviors through online trajectory optimization. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 4906–4913. IEEE, 2012. (See tassa2012.pdf). 
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc import bmatrix
from irlc import savepdf


def fz(X, a, b=None, N=None):
    """
    Helper function. Check if X is None, and if so return a list
    X = [A,A,....]
    which is N long and where each A is a (a x b) zero-matrix.
    """
    if X is not None:
        return X
    X = np.zeros((a,) if b is None else (a,b))
    return [X] * N

def LQR(A, B, d=None, Q=None, R=None, H=None, q=None, r=None, qc=None, QN=None, qN=None, qcN=None, mu=0):
    """
    Implement LQR. See (Har20, Section 2.3.1, eq.(60)-(62)) for the specific definitino of the above terms
    or the lecture slides. Note there is a typo.


    Input:
        We follow the convention A, B, etc. are lists of matrices/vectors/scalars, such that
        A_k = A[k] is the dynamical matrix, etc.

        A slight annoyance is that there are two q-terms in (Har20, eq.(60)): the vector and the scalar terms.
        we will therefore use q to refer to the vector and qc to refer to the scalar.
    Return:
        We will return the (list of) control matrices/vectors L, l such that u = l + L x
    """
    N = len(A)
    n,m = B[0].shape
    # Initialize control matrices and cost terms
    L, l = [None]*N, [None]*N
    V, v, vc = [None]*(N+1), [None]*(N+1), [None]*(N+1)
    # Initialize constant cost-function terms to zero if not specified
    QN = np.zeros((n,n)) if QN is None else QN
    qN = np.zeros((n,)) if qN is None else qN
    qcN = 0 if qcN is None else qcN
    H,q,qc,r = fz(H,m,n,N=N), fz(q,n,N=N), fz(qc,1,N=N), fz(r,m,N=N)
    d = fz(d,n, N=N)

    V[N] = QN
    v[N] = qN
    vc[N] = qcN
    # raise NotImplementedError("Initialize V[N], v[N], vc[N] here")
    
    In = np.eye(n)
    
    for k in range(N-1,-1,-1):
        # When you update S_uu and S_ux, check out (TET12, Eq (10a) and Eq (10b))
        # and remember to add regularization as the terms ... (V[k+1] + mu * In) ...
        # Note that that to find x such that
        # >>> x = A^{-1} y this
        # in a numerically stable manner this should be done as
        # >>> x = np.linalg.solve(A, y)
        # TODO: 6 lines missing.
        S_uk  = r[k] + B[k].T @ v[k+1] + B[k].T @ V[k+1] @ d[k]
        S_uu  = R[k] + B[k].T @ V[k+1] @ B[k]
        S_ux  = H[k] + B[k].T @ V[k+1] @ A[k]
        L[k]  = np.linalg.solve(-S_uu, S_ux)
        l[k]  = np.linalg.solve(-S_uu, S_uk)

        V[k]  = Q[k] + A[k].T @ V[k+1]  @ A[k] - L[k].T @ S_uu @ L[k]
        V[k]  = 0.5 * (V[k] + V[k].T)  # I recommend putting this here to keep V positive semidefinite

        v[k]  = q[k] + A[k].T @ (v[k+1] + V[k+1]@d[k]) + S_ux.T @ l[k]
        vc[k] = vc[k+1] + qc[k] + d[k].T @ v[k+1] + 1/2 * d[k].T @ V[k+1] @ d[k] + 1/2 * l[k].T @ S_uk

    return (L,l), (V,v,vc)


def dlqr_J(x,V,v,vc, QN=None, qN=None, qcN=None):
    """
    Compute cost terms. Currently not used
    """
    Jk, xN = [1/2 * V_.T @ x_ @ V_ + v_.T @ x_ + vc_ for (x_, V_,v_, vc_) in zip(x[:-1],V,v,vc)] + x[-1]
    JN = (1/2*xN.T@QN@xN if QN is not None else 0) + (qN.T@xN if qN is not None else 0)
    return sum(Jk)+ JN + (qcN if QN is not None else 0)

def lqr_rollout(x0,A,B,d,L,l):
    """
    Compute a rollout (states and actions) given solution from LQR controller function.
    """
    x, trajectory,actions = x0, [x0], []
    n,m = B[0].shape
    N = len(L)
    d = fz(d,n,1,N)
    l = fz(l,m,1,N)
    for k in range(N):
        u = L[k] @ x + l[k]
        x = A[k] @ x + B[k] @ u + d[k]
        actions.append(u)
        trajectory.append(x)
    return trajectory,actions

if __name__ ==  "__main__":
    """
    Solve this example: 
    http://cse.lab.imtlucca.it/~bemporad/teaching/ac/pdf/AC2-04-LQR-Kalman.pdf
    """
    N = 20
    A = np.ones((2,2))
    A[1,0] = 0
    B = np.asarray([[0], [1]])
    
    Q = np.zeros((2,2))
    R = np.ones((1,1))
    
    print("System matrices A, B, Q, R")
    print(bmatrix(A))  
    print(bmatrix(B))  
    print(bmatrix(Q))  
    print(bmatrix(R))  

    for rho in [0.1, 10, 100]:
        Q[0,0] = 1/rho
        (L,l), (V,v,vc) = LQR(A=[A]*N, B=[B]*N, d=None, Q=[Q]*N, R=[R]*N, QN=Q)

        x0 = np.asarray( [[1],[0]])
        trajectory, actions = lqr_rollout(x0,A=[A]*N, B=[B]*N, d=None,L=L,l=l)

        xs = np.stack(trajectory)[:,0,0]
        plt.plot(xs, 'o-', label=f'rho={rho}')

        k = 10
        print(f"Control matrix in u_k = L_k x_k + l_k at k={k}:", L[k])
    for k in [N-1,N-2,0]:
        print(f"L[{k}] is:", L[k].round(4))
    plt.title("Double integrator")
    plt.xlabel('Steps $k$')
    plt.ylabel('$x_1 = $ x[0]')
    plt.legend()
    savepdf("dlqr_double_integrator")
    plt.show()
