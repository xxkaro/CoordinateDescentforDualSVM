import numpy as np
from scipy.sparse import csr_matrix


def generate_sparse_dataset(
    n_samples: int,
    n_features: int,
    sparsity: float = 0.9,
    informative_features: int = 20,
    noise: float = 0.1,
    random_state: int = 42,
):
    """
    Generate sparse dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features (excluding target column).
    sparsity : float, optional
        Fraction of zeros in the feature matrix.
        Must be between 0 and 1.
        Higher -> more sparse.
    informative_features : int, optional
        Number of informative features (default: 20).
    noise : float, optional
        Standard deviation of noise added to the target (default: 0.1).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    X : scipy.sparse.csr_matrix
        Sparse feature matrix of shape (n_samples, n_features)

    y : np.ndarray
        Target vector with values {-1, 1}
    """

    rng = np.random.default_rng(random_state)

    density = 1.0 - sparsity
    nnz = int(n_samples * n_features * density)

    rows = np.arange(n_samples)
    cols = rng.integers(0, n_features, size=n_samples)
    data = rng.normal(size=n_samples)

    remaining = max(0, nnz - n_samples)

    rows = np.concatenate([
        rows,
        rng.integers(0, n_samples, size=remaining)
    ])

    cols = np.concatenate([
        cols,
        rng.integers(0, n_features, size=remaining)
    ])

    data = np.concatenate([
        data,
        rng.normal(size=remaining)
    ])

    X = csr_matrix((data, (rows, cols)),
                   shape=(n_samples, n_features))


    informative_idx = rng.choice(
        n_features,
        size=informative_features,
        replace=False
    )

    weights = np.zeros(n_features)

    weights[informative_idx] = rng.normal(
        loc=0,
        scale=1,
        size=informative_features
    )


    scores = X @ weights

    scores += noise * rng.normal(size=n_samples)

    y = np.where(scores > 0, 1, -1)

    return X, y