import numpy as np
from scipy.sparse import csr_matrix

def dual_coordinate_descent(
    X: csr_matrix,
    y: np.ndarray,
    U: float,
    Dii: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Dual coordinate descent for linear SVM

    Parameters
    ----------
    X : csr_matrix (l, n)
        Training instances (sparse).
    y : ndarray (l,)
        Labels in {-1, +1}.
    U : float
        Upper bound on alpha_i.
    Dii : float
        Diagonal element of D.
    max_iter : int
        Maximum number of outer iterations.
    tol : float
        Stopping tolerance on projected gradient gap: M_k - m_k < tol.
    verbose : bool
        Print progress every 10 iterations.

    Returns
    -------
    w : ndarray (n,)
        Primal weight vector.
    alpha : ndarray (l,)
        Dual variables.
    obj_history : list[float]
        Dual objective value after each outer iteration.
    """
    l, n = X.shape

    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    Q_bar_ii = np.array(X.multiply(X).sum(axis=1)).ravel() + Dii

    alpha = np.zeros(l, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)

    obj_history = []

    for k in range(max_iter):
        perm = np.random.permutation(l)

        M_k = -np.inf 
        m_k = np.inf  

        for idx in perm:
            start, end = X_indptr[idx], X_indptr[idx + 1]
            cols = X_indices[start:end]
            vals = X_data[start:end]
            yi = y[idx]

            wTx = np.dot(w[cols], vals)
            G = yi * wTx - 1.0 + Dii * alpha[idx]

            ai = alpha[idx]
            if ai == 0.0:
                PG = min(G, 0.0)
            elif ai == U:
                PG = max(G, 0.0)
            else:
                PG = G

            M_k = max(M_k, PG)
            m_k = min(m_k, PG)

            if PG == 0.0:
                continue

            old_ai = ai
            if Q_bar_ii[idx] > 0:
                new_ai = min(max(old_ai - G / Q_bar_ii[idx], 0.0), U)
            else:
                new_ai = U

            alpha[idx] = new_ai

            scale = (new_ai - old_ai) * yi
            if scale != 0.0:
                w[cols] += scale * vals

        dual_obj = 0.5 * np.dot(w, w) + 0.5 * Dii * np.dot(alpha, alpha) - np.sum(alpha)
        obj_history.append(dual_obj)

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print(f"Iter {k:4d} | dual_obj = {dual_obj:.6e} | gap = {M_k - m_k:.6e}")

        if M_k - m_k < tol:
            if verbose:
                print(f"Converged at iteration {k} (gap = {M_k - m_k:.6e} < {tol})")
            break

    return w, alpha, obj_history