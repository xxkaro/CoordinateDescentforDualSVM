import numpy as np
from scipy.sparse import csr_matrix

def load_libsvm(filepath: str) -> tuple[csr_matrix, np.ndarray]:
    """
    Reads a file in LIBSVM format and returns (X, y).

    Returns
    -------
    X : csr_matrix of shape (l, n)
        Feature matrix (sparse).
    y : ndarray of shape (l,)
        Labels in {-1, +1}.
    """
    labels = []
    indices = []
    values = []
    row_ptrs = [0]

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labels.append(int(float(parts[0])))
            count = 0
            for token in parts[1:]:
                if ":" not in token:
                    continue
                idx_str, val_str = token.split(":")
                indices.append(int(idx_str)) 
                values.append(float(val_str))
                count += 1
            row_ptrs.append(row_ptrs[-1] + count)

    y = np.array(labels, dtype=np.float64)
    y[y == 0] = -1.0

    indices = np.array(indices, dtype=np.int32)
    values = np.array(values, dtype=np.float64)
    row_ptrs = np.array(row_ptrs, dtype=np.int32)

    n = int(indices.max()) if len(indices) > 0 else 0
    indices = indices - 1 

    X = csr_matrix((values, indices, row_ptrs), shape=(len(labels), n))
    return X, y