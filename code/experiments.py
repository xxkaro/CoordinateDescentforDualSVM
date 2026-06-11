"""
Sequential experiment runner for DCD SVM project.

Phases:
  1  All DCD variants + baselines, C=1.0           
  2  Best variant x C values                      
  3  Best config on SUSY subsamples (size scaling)
  4  Best config on synthetic data (sparsity)
  5  Best config x tol values (convergence analysis)

Usage:
    python experiments.py              # running all phases sequentially
    python experiments.py 1            # running only phase 1
    python experiments.py 2 --method DCD_L2_perm   # overriding best method
    python experiments.py --summary    # showing results without running
"""

import os
import sys
import time
import json
import threading
import traceback
import numpy as np
import pandas as pd
from itertools import product
from scipy.sparse import csr_matrix

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from data_loader import load_libsvm
from data_generator import generate_sparse_dataset
from algorithm.svm import LinearSVM


SEEDS       = [42, 142, 242]
MAX_ITER    = 5_000
TIMEOUT_S   = 500         
DEFAULT_TOL = 1
TEST_RATIO  = 0.3
DATA_DIR    = "data"
RESULTS_DIR = "results"
HIST_DIR    = os.path.join(RESULTS_DIR, "histories")

DATASETS = ["a9a", "cod_rna", "news20", "rcv1", "real-sim", "skin"]

C_VALUES   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
TOL_VALUES = [1.0, 0.1, 0.01, 0.001, 0.0001]

SUSY_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SPARSITY_VALS  = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
SYNTH_N, SYNTH_D = 50_000, 5_000



def _dcd_name(loss, permute, shrinking, online):
    n = f"DCD_{loss.upper()}"
    if online:
        return n + "_online"
    if permute:
        n += "_perm"
    if shrinking:
        n += "_shrink"
    return n


def _all_dcd():
    cfgs = []
    for loss in ("l1", "l2"):
        for p, s in product((False, True), repeat=2):
            cfgs.append(dict(loss=loss, permute=p, shrinking=s, online=False))
        cfgs.append(dict(loss=loss, permute=False, shrinking=False, online=True))
    return [(cfg, _dcd_name(**cfg)) for cfg in cfgs]


def _all_baselines():
    return [
        ("LinearSVC_L1", "skl", dict(loss="hinge", dual=True)),
        ("LinearSVC_L2", "skl", dict(loss="squared_hinge", dual=True)),
        ("SVC_linear",   "svc", dict(kernel="linear", shrinking=True)),
    ]


ALL_DCD   = _all_dcd()
BASELINES = _all_baselines()


def _load(name):
    return load_libsvm(os.path.join(DATA_DIR, name))

def _split(X, y, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(X.shape[0])
    s = int((1 - TEST_RATIO) * len(idx))
    return X[idx[:s]], X[idx[s:]], y[idx[:s]], y[idx[s:]]

def _scale(Xtr, Xte):
    sc = StandardScaler(with_mean=False)
    return sc.fit_transform(Xtr), sc.transform(Xte)

def _metrics(yt, yp, ys=None):
    m = dict(
        accuracy  = accuracy_score(yt, yp),
        precision = precision_score(yt, yp, pos_label=1, zero_division=0),
        recall    = recall_score(yt, yp, pos_label=1, zero_division=0),
        f1        = f1_score(yt, yp, pos_label=1, zero_division=0),
    )
    try:
        m["auc"] = roc_auc_score(yt, ys) if ys is not None else np.nan
    except Exception:
        m["auc"] = np.nan
    return m

def _load_done(csv_path, key_cols):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path, dtype=str)
    return {tuple(str(row[c]) for c in key_cols) for _, row in df.iterrows()}

def _mk(d, kcols):
    return tuple(str(d[c]) for c in kcols)

def _save_row(csv_path, row):
    hdr = not os.path.exists(csv_path)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=hdr, index=False)

def _save_hist(ds, method, seed, C, tol, hist):
    fn = f"{ds}_{method}_C{C}_tol{tol}_seed{seed}.json"
    with open(os.path.join(HIST_DIR, fn), "w") as f:
        json.dump(hist, f)



def _timed_call(fn, timeout):
    """
    Run fn() in a daemon thread.
    Returns (result, error_str) where error_str is None on success,
    or a message on exception / timeout.
    The background thread keeps running after timeout (can't be killed),
    but the main script moves on.
    """
    box = {}

    def _worker():
        try:
            box["result"] = fn()
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        return None, f"Timeout >{timeout}s"
    if "error" in box:
        return None, str(box["error"])
    return box["result"], None

_SKIP = set()

def _load_skips(csv_path):
    """Load timed-out (dataset, method) pairs from existing CSV."""
    global _SKIP
    _SKIP = set()
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path, dtype=str)
    for _, row in df.iterrows():
        if any("Timeout" in str(v) for v in row.values):
            _SKIP.add((str(row.get("dataset", "")), str(row.get("method", ""))))


def _run_dcd(Xtr, ytr, Xte, yte, cfg, C, tol):
    mdl = LinearSVM(
        loss=cfg["loss"], C=C, max_iter=MAX_ITER, tol=tol,
        permute=cfg["permute"], shrinking=cfg["shrinking"],
        online=cfg["online"], verbose=False,
    )
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    t = time.time() - t0
    yp = mdl.predict(Xte)
    ys = mdl.decision_function(Xte)
    m  = _metrics(yte, yp, ys)
    return dict(time_s=t, n_iter=mdl.n_iter_,
                primal_obj=mdl.primal_objective(Xtr, ytr), **m), mdl.obj_history_


def _run_skl(Xtr, ytr, Xte, yte, mtype, cfg, C, tol):
    if mtype == "svc":
        mdl = SVC(C=C, max_iter=MAX_ITER, tol=tol, **cfg)
    else:
        mdl = LinearSVC(C=C, max_iter=MAX_ITER, tol=tol,
                        fit_intercept=False, **cfg)
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    t = time.time() - t0
    yp = mdl.predict(Xte)
    ys = mdl.decision_function(Xte)
    m  = _metrics(yte, yp, ys)
    ni = mdl.n_iter_
    if isinstance(ni, np.ndarray):
        ni = int(ni[0])
    return dict(time_s=t, n_iter=ni, primal_obj=np.nan, **m), None


def _try_run(csv_path, done, kcols, kd, Xtr, ytr, Xte, yte,
             mtype, mname, mcfg, C, tol, ds_label):
    k = _mk(kd, kcols)
    if k in done:
        return
    
    tag = f"{ds_label} | {mname} | C={C} | tol={tol}"
    print(f"    {tag} ... ", end="", flush=True)
    if (ds_label, mname) in _SKIP:
        print(f"    {tag} ... SKIP (prior timeout)")
        row = {**kd, "error": "Skipped (prior timeout)"}
        _save_row(csv_path, row)
        done.add(k)
        return

    if mtype == "dcd":
        fn = lambda: _run_dcd(Xtr, ytr, Xte, yte, mcfg, C, tol)
    else:
        fn = lambda: _run_skl(Xtr, ytr, Xte, yte, mtype, mcfg, C, tol)

    result, err = _timed_call(fn, TIMEOUT_S)

    if err:
        print(f"FAIL: {err}")
        if "Timeout" in err:
            _SKIP.add((ds_label, mname))
        row = {**kd, "error": err}
    else:
        res, hist = result if mtype == "dcd" else (result[0], None)
        if mtype == "dcd" and hist is not None:
            _save_hist(ds_label, mname, kd["seed"], C, tol, hist)
        row = {**kd, **res}
        print(f"{res['time_s']:.1f}s  acc={res['accuracy']:.4f}  iter={res['n_iter']}")

    _save_row(csv_path, row)
    done.add(k)



def _pick_best(csv_path, group_col, metric="accuracy", only_dcd=True):
    df = pd.read_csv(csv_path)
    df = df[pd.to_numeric(df["accuracy"], errors="coerce").notna()]
    if only_dcd:
        df = df[df["method"].str.startswith("DCD_")]
    grouped = df.groupby(group_col)[metric].mean()
    best = grouped.idxmax()
    print(f"\n  Auto-pick: best {group_col} = {best}"
          f"  (mean {metric} = {grouped[best]:.4f})")
    print(grouped.sort_values(ascending=False).to_string())
    return best



def phase_1():
    print("\n" + "=" * 65)
    print("  PHASE 1: All methods x C=1.0 -> pick best DCD variant")
    print("=" * 65)
    csv, kcols = os.path.join(RESULTS_DIR, "phase_1.csv"), ["dataset", "method", "seed"]
    done = _load_done(csv, kcols)
    _load_skips(csv)
    C, tol = 1.0, DEFAULT_TOL
    for ds in DATASETS:
        X, y = None, None
        for seed in SEEDS:
            pending = []
            for cfg, mname in ALL_DCD:
                kd = dict(dataset=ds, method=mname, seed=seed, C=C, tol=tol)
                if _mk(kd, kcols) not in done:
                    pending.append(("dcd", mname, cfg, kd))
            for bname, btype, bcfg in BASELINES:
                kd = dict(dataset=ds, method=bname, seed=seed, C=C, tol=tol)
                if _mk(kd, kcols) not in done:
                    pending.append((btype, bname, bcfg, kd))
            if not pending:
                continue
            if X is None:
                print(f"\n  Loading {ds} ...")
                X, y = _load(ds)
                print(f"    {X.shape[0]:,} x {X.shape[1]:,}, {X.nnz:,} nnz")
            Xtr, Xte, ytr, yte = _split(X, y, seed)
            Xtr, Xte = _scale(Xtr, Xte)
            for mt, mn, mc, kd in pending:
                _try_run(csv, done, kcols, kd, Xtr, ytr, Xte, yte, mt, mn, mc, C, tol, ds)
    return _pick_best(csv, "method")


def phase_2(best_method):
    print("\n" + "=" * 65)
    print(f"  PHASE 2: {best_method} x C values -> pick best C")
    print("=" * 65)
    csv, kcols = os.path.join(RESULTS_DIR, "phase_2.csv"), ["dataset", "method", "seed", "C"]
    done = _load_done(csv, kcols)
    _load_skips(csv)
    tol = DEFAULT_TOL
    dcd_cfg = next(cfg for cfg, name in ALL_DCD if name == best_method)
    methods = [("dcd", best_method, dcd_cfg)] + [(bt, bn, bc) for bn, bt, bc in BASELINES]
    for ds in DATASETS:
        X, y = None, None
        for C in C_VALUES:
            for seed in SEEDS:
                pending = [(mt, mn, mc, dict(dataset=ds, method=mn, seed=seed, C=C, tol=tol))
                           for mt, mn, mc in methods
                           if _mk(dict(dataset=ds, method=mn, seed=seed, C=C), kcols) not in done]
                if not pending:
                    continue
                if X is None:
                    print(f"\n  Loading {ds} ...")
                    X, y = _load(ds)
                    print(f"    {X.shape[0]:,} x {X.shape[1]:,}")
                Xtr, Xte, ytr, yte = _split(X, y, seed)
                Xtr, Xte = _scale(Xtr, Xte)
                for mt, mn, mc, kd in pending:
                    _try_run(csv, done, kcols, kd, Xtr, ytr, Xte, yte, mt, mn, mc, C, tol, ds)
    return float(_pick_best(csv, "C"))


def phase_3(best_method, best_C):
    print("\n" + "=" * 65)
    print(f"  PHASE 3: SUSY size scaling - {best_method}, C={best_C}")
    print("=" * 65)
    csv, kcols = os.path.join(RESULTS_DIR, "phase_3.csv"), ["dataset", "method", "seed", "fraction"]
    done = _load_done(csv, kcols)
    _load_skips(csv)
    dcd_cfg = next(cfg for cfg, name in ALL_DCD if name == best_method)
    methods = [("dcd", best_method, dcd_cfg)] + [(bt, bn, bc) for bn, bt, bc in BASELINES]
    any_pending = any(
        _mk(dict(dataset=f"SUSY_{int(f*100)}pct", method=mn, seed=s, fraction=f), kcols) not in done
        for f in SUSY_FRACTIONS for _, mn, _ in methods for s in SEEDS
    )
    if not any_pending:
        print("  All done.")
        return
    print(f"\n  Loading SUSY ...")
    X_full, y_full = _load("SUSY")
    print(f"    {X_full.shape[0]:,} x {X_full.shape[1]:,}")
    for frac in SUSY_FRACTIONS:
        n = int(frac * X_full.shape[0])
        ds_label = f"SUSY_{int(frac*100)}pct"
        print(f"\n  Subsample: {ds_label} ({n:,})")
        idx = np.random.RandomState(0).choice(X_full.shape[0], size=n, replace=False)
        Xsub, ysub = X_full[idx], y_full[idx]
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = _split(Xsub, ysub, seed)
            Xtr, Xte = _scale(Xtr, Xte)
            for mt, mn, mc in methods:
                kd = dict(dataset=ds_label, method=mn, seed=seed,
                          C=best_C, tol=DEFAULT_TOL, fraction=frac)
                _try_run(csv, done, kcols, kd, Xtr, ytr, Xte, yte,
                         mt, mn, mc, best_C, DEFAULT_TOL, ds_label)


def phase_4(best_method, best_C):
    print("\n" + "=" * 65)
    print(f"  PHASE 4: Synthetic sparsity - {best_method}, C={best_C}")
    print("=" * 65)
    csv, kcols = os.path.join(RESULTS_DIR, "phase_4.csv"), ["dataset", "method", "seed", "sparsity"]
    done = _load_done(csv, kcols)
    _load_skips(csv)
    dcd_cfg = next(cfg for cfg, name in ALL_DCD if name == best_method)
    methods = [("dcd", best_method, dcd_cfg)] + [(bt, bn, bc) for bn, bt, bc in BASELINES]
    for sp in SPARSITY_VALS:
        ds_label = f"synth_sp{sp}"
        print(f"\n  Generating {ds_label} ({SYNTH_N:,}x{SYNTH_D:,}) ...")
        X, y = generate_sparse_dataset(n_samples=SYNTH_N, n_features=SYNTH_D,
                                        sparsity=sp, random_state=0)
        print(f"    nnz={X.nnz:,}")
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = _split(X, y, seed)
            Xtr, Xte = _scale(Xtr, Xte)
            for mt, mn, mc in methods:
                kd = dict(dataset=ds_label, method=mn, seed=seed,
                          C=best_C, tol=DEFAULT_TOL, sparsity=sp)
                _try_run(csv, done, kcols, kd, Xtr, ytr, Xte, yte,
                         mt, mn, mc, best_C, DEFAULT_TOL, ds_label)


def phase_5(best_method, best_C):
    print("\n" + "=" * 65)
    print(f"  PHASE 5: Convergence analysis - {best_method}, C={best_C} x tol")
    print("=" * 65)
    csv, kcols = os.path.join(RESULTS_DIR, "phase_5.csv"), ["dataset", "method", "seed", "tol"]
    done = _load_done(csv, kcols)
    _load_skips(csv)
    dcd_cfg = next(cfg for cfg, name in ALL_DCD if name == best_method)
    for ds in DATASETS:
        X, y = None, None
        for tol in TOL_VALUES:
            for seed in SEEDS:
                kd = dict(dataset=ds, method=best_method, seed=seed, C=best_C, tol=tol)
                if _mk(kd, kcols) in done:
                    continue
                if X is None:
                    print(f"\n  Loading {ds} ...")
                    X, y = _load(ds)
                    print(f"    {X.shape[0]:,} x {X.shape[1]:,}")
                Xtr, Xte, ytr, yte = _split(X, y, seed)
                Xtr, Xte = _scale(Xtr, Xte)
                _try_run(csv, done, kcols, kd, Xtr, ytr, Xte, yte,
                         "dcd", best_method, dcd_cfg, best_C, tol, ds)

def show_summary():
    group_cols = {1: "method", 2: "C", 3: "fraction", 4: "sparsity", 5: "tol"}
    for i in range(1, 6):
        csv = os.path.join(RESULTS_DIR, f"phase_{i}.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        df = df[pd.to_numeric(df["accuracy"], errors="coerce").notna()]
        print(f"\n{'='*65}\n  Phase {i}: {len(df)} runs\n{'='*65}")
        gc = group_cols[i]
        if gc in df.columns:
            print(df.groupby(gc).agg(
                accuracy=("accuracy", "mean"), f1=("f1", "mean"),
                time_s=("time_s", "mean"), n_iter=("n_iter", "mean"),
            ).round(4).to_string())



def _read_best_method():
    csv = os.path.join(RESULTS_DIR, "phase_1.csv")
    return _pick_best(csv, "method") if os.path.exists(csv) else None

def _read_best_C():
    csv = os.path.join(RESULTS_DIR, "phase_2.csv")
    return float(_pick_best(csv, "C")) if os.path.exists(csv) else None



if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HIST_DIR, exist_ok=True)

    args = sys.argv[1:]

    if "--summary" in args:
        show_summary()
        sys.exit(0)

    override_method = None
    if "--method" in args:
        idx = args.index("--method")
        override_method = args[idx + 1]
        args = [a for a in args if a not in ("--method", override_method)]

    requested = [int(a) for a in args if a.isdigit()]
    if not requested:
        requested = [1, 2, 3, 4, 5]

    best_method = override_method or _read_best_method()
    best_C      = _read_best_C()

    for phase_num in requested:
        if phase_num == 1:
            best_method = phase_1()
        elif phase_num == 2:
            if best_method is None:
                print("Run phase 1 first (or use --method).")
                continue
            best_C = phase_2(best_method)
        elif phase_num in (3, 4, 5):
            if None in (best_method, best_C):
                print("Run phases 1-2 first.")
                continue
            [phase_3, phase_4, phase_5][phase_num - 3](best_method, best_C)

    print(f"\n  method={best_method}  C={best_C}  tol={DEFAULT_TOL}")