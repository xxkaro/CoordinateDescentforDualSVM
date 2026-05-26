import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from data_loader import load_libsvm
from svm import LinearSVM


def main():
    data_path = "data/news20" 
    print(f"Loading {data_path} ...")
    X, y = load_libsvm(data_path)
    print(f"  X: {X.shape}, nnz = {X.nnz}")
    print(f"  y: {y.shape}, classes = {np.unique(y)}")

    np.random.seed(42)
    l = X.shape[0]
    perm = np.random.permutation(l)
    split = int(0.8 * l)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    C = 1.0

    print("\n=== DCD L1-SVM ===")
    model_l1 = LinearSVM(loss="l1", C=C, max_iter=5000, tol=1e-4, permute=True, verbose=False)
    t0 = time.time()
    model_l1.fit(X_train, y_train)
    t_l1 = time.time() - t0
    acc_l1 = model_l1.score(X_test, y_test)
    pobj_l1 = model_l1.primal_objective(X_train, y_train)
    print(f"  Time: {t_l1:.3f}s  | Iter: {model_l1.n_iter_} | Acc: {acc_l1:.4f} | Primal obj: {pobj_l1:.6f}")

    print("\n=== DCD L2-SVM ===")
    model_l2 = LinearSVM(loss="l2", C=C, max_iter=5000, tol=1e-4, permute=True, verbose=False)
    t0 = time.time()
    model_l2.fit(X_train, y_train)
    t_l2 = time.time() - t0
    acc_l2 = model_l2.score(X_test, y_test)
    pobj_l2 = model_l2.primal_objective(X_train, y_train)
    print(f"  Time: {t_l2:.3f}s | Iter: {model_l2.n_iter_} | Acc: {acc_l2:.4f} | Primal obj: {pobj_l2:.6f}")

    print("\n=== sklearn LinearSVC (L1 hinge) ===")
    skl_l1 = LinearSVC(loss="hinge", C=C, max_iter=5000, dual=True)
    t0 = time.time()
    skl_l1.fit(X_train, y_train)
    t_skl_l1 = time.time() - t0
    acc_skl_l1 = accuracy_score(y_test, skl_l1.predict(X_test))
    print(f"  Time: {t_skl_l1:.3f}s | Iter: {skl_l1.n_iter_} | Acc: {acc_skl_l1:.4f}")

    print("\n=== sklearn LinearSVC (L2 hinge) ===")
    skl_l2 = LinearSVC(loss="squared_hinge", C=C, max_iter=5000, dual=True)
    t0 = time.time()
    skl_l2.fit(X_train, y_train)
    t_skl_l2 = time.time() - t0
    acc_skl_l2 = accuracy_score(y_test, skl_l2.predict(X_test))
    print(f"  Time: {t_skl_l2:.3f}s | Iter: {skl_l2.n_iter_} | Acc: {acc_skl_l2:.4f}")

    print("\n=== Summary ===")
    print(f"{'Method':<25} {'Time (s)':>10} {'Iter':>10} {'Accuracy':>10}")
    print("-" * 57)
    print(f"{'DCD L1-SVM':<25} {t_l1:>10.3f} {model_l1.n_iter_:>10} {acc_l1:>10.4f}")
    print(f"{'DCD L2-SVM':<25} {t_l2:>10.3f} {model_l2.n_iter_:>10} {acc_l2:>10.4f}")
    print(f"{'sklearn LinearSVC L1':<25} {t_skl_l1:>10.3f} {skl_l1.n_iter_:>10} {acc_skl_l1:>10.4f}")
    print(f"{'sklearn LinearSVC L2':<25} {t_skl_l2:>10.3f} {skl_l2.n_iter_:>10} {acc_skl_l2:>10.4f}")

if __name__ == "__main__":
    main()