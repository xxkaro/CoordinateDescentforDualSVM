import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.preprocessing import StandardScaler

from data_loader import load_libsvm
from data_generator import generate_sparse_dataset
from algorithm.svm import LinearSVM


def main():
    data_path = "data/a9a"
    print(f"Loading {data_path} ...")
    # X, y = load_libsvm(data_path)
    X, y = generate_sparse_dataset(
        n_samples=200000,
        n_features=500,
        sparsity=0.9,
        random_state=42
    )
    print(f"  X: {X.shape}, nnz = {X.nnz}")
    print(f"  y: {y.shape}, classes = {np.unique(y)}")
 
    np.random.seed(42)
    l = X.shape[0]
    perm = np.random.permutation(l)
    split = int(0.8 * l)
    train_idx, test_idx = perm[:split], perm[split:]
 
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    C = 1
    MAX_ITER = 400000
    TOL = 1
 
    results = []

    for loss, do_permute, do_shrink, do_online in product(
        ["l1", "l2"], [False, True], [False, True], [False, True]
    ):
        label = f"DCD {loss.upper()} perm={do_permute:<5} shrink={do_shrink} online={do_online}"
        print(f"\n=== {label} ===")
 
        model = LinearSVM(
            loss=loss, C=C, max_iter=MAX_ITER, tol=TOL,
            permute=do_permute, shrinking=do_shrink, verbose=False, online=do_online
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
 
        acc = model.score(X_test, y_test)
        pobj = model.primal_objective(X_train, y_train)
        print(f"  Time: {elapsed:.3f}s | Iter: {model.n_iter_} | Acc: {acc:.4f} | Primal obj: {pobj:.6f}")
 
        results.append((label, elapsed, model.n_iter_, acc, pobj))
 
    # for skl_loss, skl_label in [("hinge", "L1"), ("squared_hinge", "L2")]:
    label = f"sklearn LinearSVC"
    print(f"\n=== {label} ===")

    skl = SVC(kernel="linear", shrinking=True, C=C, max_iter=MAX_ITER, tol=TOL)
    t0 = time.time()
    skl.fit(X_train, y_train)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, skl.predict(X_test))
    print(f"  Time: {elapsed:.3f}s | Iter: {skl.n_iter_} | Acc: {acc:.4f}")

    results.append((label, elapsed, skl.n_iter_, acc, None))

    print("\n" + "=" * 85)
    print(f"{'Method':<45} {'Time':>8} {'Iter':>6} {'Acc':>8} {'Primal obj':>14}")
    print("-" * 85)
    for label, t, it, acc, pobj in results:
        if type(it) is np.ndarray or type(it) is list:
            it = it[0] if len(it) > 0 else 0
        pobj_str = f"{pobj:.4f}" if pobj is not None else "-"
        print(f"{label:<45} {t:>8.3f} {it:>6} {acc:>8.4f} {pobj_str:>14}")


if __name__ == "__main__":
    main()