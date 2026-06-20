import numpy as np
from scipy.sparse import csr_matrix, issparse

from algorithm.coordinate_descent import dual_coordinate_descent
from algorithm.losses import LOSS_REGISTRY

class LinearSVM:
    """
    Linear SVM trained via dual coordinate descent.

    Parameters
    ----------
    loss : str
        'l1' for hinge loss (L1-SVM) or 'l2' for squared hinge (L2-SVM).
    C : float
        Penalty parameter.
    max_iter : int
        Maximum outer iterations for DCD.
    tol : float
        Stopping tolerance.
    permute : bool
        Whether to permute coordinates in each outer iteration.
    verbose : bool
        Print convergence info.
    """

    def __init__(
        self,
        loss: str = "l1",
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        permute: bool = True,
        shrinking: bool = False,
        online: bool = False,
        verbose: bool = False
    ):
        if loss not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss {loss!r}. Available: {list(LOSS_REGISTRY)}")

        self.loss_name = loss
        self.loss_fn = LOSS_REGISTRY[loss]() 
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.permute = permute
        self.shrinking = shrinking
        self.online = online
        self.verbose = verbose
        self.n_iter_ = 0

        self.w_ = None
        self.alpha_ = None
        self.obj_history_ = None

    def fit(self, X, y):
        """
        Fits the model on training data.

        Parameters
        ----------
        X : array-like or csr_matrix of shape (l, n)
        y : array-like of shape (l,), labels in {-1, +1}
        """
        if not issparse(X):
            X = csr_matrix(X)

        y = np.asarray(y, dtype=np.float64).ravel()
        assert set(np.unique(y)).issubset({-1.0, 1.0}), "Labels must be {-1, +1}"

        U, Dii = self.loss_fn.dual_params(self.C)

        self.w_, self.alpha_, self.history_ = dual_coordinate_descent(
            X, y,
            U=U,
            Dii=Dii,
            max_iter=self.max_iter,
            tol=self.tol,
            permute=self.permute,
            shrinking=self.shrinking,
            online=self.online,
            verbose=self.verbose,
        )
        self.obj_history_ = self.history_["dual_obj"]
        self.n_iter_ = len(self.obj_history_)
        return self

    def decision_function(self, X):
        """Returns w^T x for each instance."""
        if not issparse(X):
            X = csr_matrix(X)
        return X.dot(self.w_)

    def predict(self, X):
        """Predicts labels {-1, +1}."""
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def score(self, X, y):
        """Classification accuracy."""
        y = np.asarray(y, dtype=np.float64).ravel()
        return np.mean(self.predict(X) == y)

    def primal_objective(self, X, y):
        """
        Computes primal objective: (1/2)||w||^2 + C * loss(w; X, y).
        """
        if not issparse(X):
            X = csr_matrix(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        reg = 0.5 * np.dot(self.w_, self.w_)
        margins = y * X.dot(self.w_)
        loss_val = self.C * self.loss_fn.primal_loss(margins).sum()

        return reg + loss_val