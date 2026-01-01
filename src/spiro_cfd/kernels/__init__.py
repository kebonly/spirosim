import numpy as np

def stokeslet_regularized(x, x0, f, mu=1.0, eps=0.05):
    """
    x:  (N,3) evaluation points
    x0: (3,)  stokeslet location
    f:  (3,)  force vector
    returns u: (N,3)
    """
    r = x - x0[None, :]
    r2 = np.sum(r*r, axis=1)                 # (N,)
    denom = (r2 + eps**2)**1.5               # (N,)
    # Build G_ij for each point: (N,3,3)
    I = np.eye(3)[None, :, :]
    rr = r[:, :, None] * r[:, None, :]       # outer products (N,3,3)
    G = ((r2 + 2*eps**2)[:, None, None]*I + rr) / denom[:, None, None]
    G *= 1.0 / (8*np.pi*mu)
    u = np.einsum('nij,j->ni', G, f)
    return u
