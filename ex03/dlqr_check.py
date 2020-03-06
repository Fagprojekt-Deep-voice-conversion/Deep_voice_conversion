"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import numpy as np
from irlc.ex03.dlqr import LQR

def urnd(sz):
    return np.random.uniform(-1, 1, sz)

if __name__ == "__main__":
    np.random.seed(42)
    n,m,N = 3,2,4
    """
    Create a randomized, nonsense control problem and solve it. Since seed is fixed we can expect same solution.
    """
    # system tersm
    A = [urnd((n, n)) for _ in range(N)]
    B = [urnd((n, m)) for _ in range(N)]
    d = [urnd((n,)) for _ in range(N)]
    # cost terms
    Q = [urnd((n, n)) for _ in range(N)]    
    R = [urnd((m, m)) for _ in range(N)]
    H = [urnd((m, n)) for _ in range(N)]
    q = [urnd((n,)) for _ in range(N)]
    qc = [urnd(()) for _ in range(N)]
    r = [urnd((m,)) for _ in range(N)]
    # terminal costs
    QN = urnd((n, n))
    qN = urnd((n,))
    qcN = urnd(())

    (L, l), (V, v, vc) = LQR(A=A, B=B, d=d, Q=Q, R=R, H=H, q=q, r=r, qc=qc, QN=QN, qN=qN, qcN=qcN, mu=0)

    print(", ".join([f"l[{k}]={l[k].round(4)}" for k in [N - 1, N - 2, 0]]))  
    print("\n".join([f"L[{k}]={L[k].round(4)}" for k in [N - 1, N - 2, 0]]))

    print("\n".join([f"V[{k}]={V[k].round(4)}" for k in [0]]))
    print(", ".join([f"v[{k}]={v[k].round(4)}" for k in [N, N - 1, 0]]))
    print(", ".join([f"vc[{k}]={vc[k].round(4)}" for k in [N, N - 1, 0]]))  
