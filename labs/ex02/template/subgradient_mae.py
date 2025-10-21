import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    N = y.shape[0]
    e = y - tx.dot(w)

    t_mod = tx.copy()
    
    t_mod[e < 0] = -tx[e < 0]
    t_mod[e == 0] = 0

    if not e.any():
        print("Error 0 encountered")
    
    grad = 1/N * np.sum(-t_mod, axis=0)

    return grad
