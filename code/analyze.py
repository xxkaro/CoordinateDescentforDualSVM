import os
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
METRICS     = ["accuracy", "f1", "auc", "time_s", "n_iter"]


def load(phase):
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"phase_{phase}.csv"))
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df[df["accuracy"].notna()].copy()


def save(df, name):
    path = os.path.join(TABLES_DIR, name)
    df.to_csv(path)
    print(f"  -> {path}")
    return df


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


os.makedirs(TABLES_DIR, exist_ok=True)


header("PHASE 1: Method comparison (C=1.0)")
p1 = load(1)

t1a = (p1.groupby("method")[["accuracy", "f1", "auc", "time_s", "n_iter"]]
         .mean()
         .sort_values("accuracy", ascending=False)
         .round(4))
print("\n[1a] Method ranking — mean over 6 datasets x 3 seeds")
print(t1a.to_string())
save(t1a, "1a_method_ranking.csv")

acc1b = (p1.groupby(["method", "dataset"])["accuracy"]
           .mean()
           .round(4)
           .unstack("dataset")
           .sort_values("a9a", ascending=False))
acc1b.index.name = "method"
print("\n[1b] Accuracy per method x dataset")
print(acc1b.to_string())
save(acc1b, "1b_accuracy_method_x_dataset.csv")

# 1c. DCD variants only
dcd = p1[p1["method"].str.startswith("DCD_")]
t1c = (dcd.groupby("method")[["accuracy", "f1", "auc", "time_s", "n_iter"]]
          .mean()
          .sort_values("accuracy", ascending=False)
          .round(4))
print("\n[1c] DCD variants ranking")
print(t1c.to_string())
save(t1c, "1c_dcd_variants_ranking.csv")

# 1d. Stability across seeds (std)
t1d = (p1.groupby("method")["accuracy"]
         .agg(["mean", "std", "min", "max"])
         .sort_values("mean", ascending=False)
         .round(4))
t1d.columns = ["acc_mean", "acc_std", "acc_min", "acc_max"]
print("\n[1d] Stability (accuracy std over seeds x datasets)")
print(t1d.to_string())
save(t1d, "1d_stability.csv")


header("PHASE 2: Impact of penalty parameter C")
p2 = load(2)

t2a = (p2.groupby(["method", "C"])["accuracy"]
         .mean()
         .round(4)
         .unstack("C"))
print("\n[2a] Accuracy vs C (mean over datasets x seeds)")
print(t2a.to_string())
save(t2a, "2a_accuracy_vs_C.csv")

# 2b. Training time vs C
t2b = (p2.groupby(["method", "C"])["time_s"]
         .mean()
         .round(3)
         .unstack("C"))
print("\n[2b] Training time [s] vs C")
print(t2b.to_string())
save(t2b, "2b_time_vs_C.csv")

for method in sorted(p2["method"].unique()):
    sub = p2[p2["method"] == method]
    best = (sub.groupby(["dataset", "C"])["accuracy"]
               .mean()
               .round(4)
               .unstack("C"))
    save(best, f"2c_accuracy_vs_C_{method}.csv")
print("\n[2c] Accuracy vs C per dataset - saved separately per method")

t2d = (p2.groupby(["method", "C"])["n_iter"]
         .mean()
         .round(1)
         .unstack("C"))
print("\n[2d] Iterations vs C")
print(t2d.to_string())
save(t2d, "2d_iters_vs_C.csv")



header("PHASE 3: Scalability - SUSY subsamples")
p3 = load(3)

t3a = (p3.groupby(["method", "fraction"])["accuracy"]
         .mean()
         .round(4)
         .unstack("fraction"))
print("\n[3a] Accuracy vs SUSY fraction")
print(t3a.to_string())
save(t3a, "3a_accuracy_vs_fraction.csv")

t3b = (p3.groupby(["method", "fraction"])["time_s"]
         .mean()
         .round(2)
         .unstack("fraction"))
print("\n[3b] Training time [s] vs SUSY fraction")
print(t3b.to_string())
save(t3b, "3b_time_vs_fraction.csv")

t3c = (p3.groupby(["method", "fraction"])[["accuracy", "time_s", "n_iter"]]
         .mean()
         .round(4))
save(t3c, "3c_full_vs_fraction.csv")
print("\n[3c] Full table (accuracy + time + iter) saved")



header("PHASE 4: Impact of sparsity - synthetic data")
p4 = load(4)

t4a = (p4.groupby(["method", "sparsity"])["accuracy"]
         .mean()
         .round(4)
         .unstack("sparsity"))
print("\n[4a] Accuracy vs sparsity")
print(t4a.to_string())
save(t4a, "4a_accuracy_vs_sparsity.csv")

t4b = (p4.groupby(["method", "sparsity"])["time_s"]
         .mean()
         .round(3)
         .unstack("sparsity"))
print("\n[4b] Training time [s] vs sparsity")
print(t4b.to_string())
save(t4b, "4b_time_vs_sparsity.csv")

t4c = (p4.groupby(["method", "sparsity"])["f1"]
         .mean()
         .round(4)
         .unstack("sparsity"))
print("\n[4c] F1 vs sparsity")
print(t4c.to_string())
save(t4c, "4c_f1_vs_sparsity.csv")


header("PHASE 5: Convergence analysis - impact of tol")
p5 = load(5)

t5a = (p5.groupby(["dataset", "tol"])["n_iter"]
         .mean()
         .round(1)
         .unstack("tol"))
t5a.loc["MEAN"] = t5a.mean().round(1)
print("\n[5a] Iterations vs tol")
print(t5a.to_string())
save(t5a, "5a_iters_vs_tol.csv")

t5b = (p5.groupby(["dataset", "tol"])["time_s"]
         .mean()
         .round(3)
         .unstack("tol"))
t5b.loc["MEAN"] = t5b.mean().round(3)
print("\n[5b] Training time [s] vs tol")
print(t5b.to_string())
save(t5b, "5b_time_vs_tol.csv")

t5c = (p5.groupby(["dataset", "tol"])["accuracy"]
         .mean()
         .round(4)
         .unstack("tol"))
t5c.loc["MEAN"] = t5c.mean().round(4)
print("\n[5c] Accuracy vs tol")
print(t5c.to_string())
save(t5c, "5c_accuracy_vs_tol.csv")


header("SUMMARY: data completeness")
for i in range(1, 6):
    path = os.path.join(RESULTS_DIR, f"phase_{i}.csv")
    if not os.path.exists(path):
        continue
    df_raw = pd.read_csv(path)
    valid   = pd.to_numeric(df_raw["accuracy"], errors="coerce").notna().sum()
    total   = len(df_raw)
    timeout = df_raw.apply(
        lambda r: any("Timeout" in str(v) or "Skip" in str(v) for v in r.values), axis=1
    ).sum()
    print(f"  phase_{i}: {valid}/{total} valid  ({timeout} timeout/skip)")

print(f"\nAll tables saved to: {TABLES_DIR}/")