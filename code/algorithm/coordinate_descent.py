import time
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from numba import njit

@njit(fastmath=True)
def _dcd_inner(X_data, X_indices, X_indptr, y, alpha, w, Q_bar_ii,
               Dii, U, perm, active, M_bar, m_bar, do_shrink):
    M_k = -np.inf
    m_k = np.inf
 
    for p in range(len(perm)):
        idx = perm[p]
 
        if not active[idx]:
            continue
 
        start, end = X_indptr[idx], X_indptr[idx + 1]
        yi = y[idx]
        ai = alpha[idx]
 
        # w^T x_i  (eq. 12)
        wTx = 0.0
        for j in range(start, end):
            wTx += w[X_indices[j]] * X_data[j]
 
        G = yi * wTx - 1.0 + Dii * ai
 
        # Shrinking
        if len(perm) > 1 and do_shrink:
            if ai == 0.0 and G > M_bar:
                active[idx] = False
                continue
            if ai == U and G < m_bar:
                active[idx] = False
                continue
 
        # Projected gradient (eq. 8)
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
 
        # Update alpha_i
        old_ai = ai
        if Q_bar_ii[idx] > 0:
            new_ai = min(max(old_ai - G / Q_bar_ii[idx], 0.0), U)
        else:
            new_ai = U
 
        alpha[idx] = new_ai
 
        # Maintain w
        scale = (new_ai - old_ai) * yi
        if scale != 0.0:
            for j in range(start, end):
                w[X_indices[j]] += scale * X_data[j]
 
    return M_k, m_k


def dual_coordinate_descent(
    X: csr_matrix,
    y: np.ndarray,
    U: float,
    Dii: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
    permute: bool = True,
    online: bool = False,
    shrinking: bool = False,
    verbose: bool = False,
    alpha_star: np.ndarray = None,
    f_star: float = None,
) -> tuple[np.ndarray, np.ndarray, dict]:

    l, n = X.shape
 
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)
 
    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    online_seen = 0
    online_M = -np.inf
    online_m = np.inf
 
    Q_bar_ii = np.array(X.multiply(X).sum(axis=1)).ravel() + Dii
 
    alpha = np.zeros(l, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)
 
    history = {"dual_obj": [], "gap": []}
    if alpha_star is not None:
        history["alpha_dist"] = []
    if f_star is not None:
        history["subopt"] = []
 
    pbar = tqdm(range(max_iter), desc="DCD", leave=True)
    perm = np.arange(l, dtype=np.int64)
 
    # Shrinking state
    active = np.ones(l, dtype=np.bool_)
    M_bar = np.inf
    m_bar = -np.inf
 
    for k in pbar:
        if online:
            perm = np.random.randint(0, l, size=1)
        elif permute:
            np.random.shuffle(perm)
 
        M_k, m_k = _dcd_inner(
            X_data, X_indices, X_indptr, y, alpha, w, Q_bar_ii,
            Dii, U, perm, active, M_bar, m_bar, shrinking
        )
        if online:
            online_M = max(online_M, M_k)
            online_m = min(online_m, m_k)
            online_seen += len(perm)

        dual_obj = 0.5 * np.dot(w, w) + 0.5 * Dii * np.dot(alpha, alpha) - np.sum(alpha)
        history["dual_obj"].append(dual_obj)
        history["gap"].append(M_k - m_k)
        if alpha_star is not None:
            history["alpha_dist"].append(float(np.linalg.norm(alpha - alpha_star)))
        if f_star is not None:
            history["subopt"].append(dual_obj - f_star)
 
        if verbose and (k % 10 == 0 or k == max_iter - 1):
            n_active = int(np.sum(active))
            pbar.set_postfix({
                "obj": f"{dual_obj:.6f}",
                "gap": f"{M_k - m_k:.6e}",
                "active": f"{n_active}/{l}",
            })
 
        # Stopping / unshrinking logic
        if online:
            if online_seen >= l:
                if online_M - online_m < tol:
                    if verbose:
                        pbar.write(
                            f"Online converged at iteration {k}: "
                            f"M_k - m_k = {online_M - online_m:.6e} < tol"
                        )
                    break

                online_seen = 0
                online_M = -np.inf
                online_m = np.inf


        else:
            if M_k - m_k < tol:
                if shrinking and not np.all(active):
                    # Active set converged but some elements are shrunken.
                    # Unshrinking
                    active[:] = True
                    M_bar = np.inf
                    m_bar = -np.inf
                    if verbose:
                        pbar.write(f"Iter {k}: active set converged, unshrinking...")
                    continue
                else:
                    # Either shrinking is off, or all elements are active
                    # and we still converged.
                    if verbose:
                        pbar.write(
                            f"Converged at iteration {k}: "
                            f"M_k - m_k = {M_k - m_k:.6e} < tol"
                        )
                    break
 
        # Update shrinking thresholds for next iteration
        if shrinking:
            M_bar = M_k if M_k > 0 else np.inf
            m_bar = m_k if m_k < 0 else -np.inf
 
    return w, alpha, history
