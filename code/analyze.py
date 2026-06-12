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


def mean_std_str(series):
    """Format as 'mean ± std'."""
    return f"{series.mean():.4f} ± {series.std():.4f}"


def pivot_mean_std(df, group_cols, val_col, pivot_col):
    """Create a pivot with 'mean ± std' strings."""
    agg = df.groupby(group_cols)[val_col].agg(["mean", "std"])
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    return agg["cell"].unstack(pivot_col)


os.makedirs(TABLES_DIR, exist_ok=True)

header("PHASE 1: Method comparison (C=1.0)")
p1 = load(1)

# 1a. Method ranking — sorted by AUC, with std
t1a_mean = (p1.groupby("method")[["accuracy", "f1", "auc", "time_s", "n_iter"]]
              .mean().round(4))
t1a_std  = (p1.groupby("method")[["accuracy", "f1", "auc", "time_s", "n_iter"]]
              .std().round(4))
t1a_std.columns = [c + "_std" for c in t1a_std.columns]
t1a = pd.concat([t1a_mean, t1a_std], axis=1)
# Reorder columns: metric, metric_std, ...
col_order = []
for m in ["accuracy", "f1", "auc", "time_s", "n_iter"]:
    col_order += [m, m + "_std"]
t1a = t1a[col_order].sort_values("auc", ascending=False)
print("\n[1a] Method ranking — mean ± std over 6 datasets x 3 seeds (sorted by AUC)")
print(t1a.to_string())
save(t1a, "1a_method_ranking.csv")

# 1b. Accuracy per method x dataset
acc1b = (p1.groupby(["method", "dataset"])["accuracy"]
           .mean()
           .round(4)
           .unstack("dataset"))
# Sort by mean AUC across datasets
method_auc_order = t1a.index.tolist()
acc1b = acc1b.reindex(method_auc_order)
acc1b.index.name = "method"
print("\n[1b] Accuracy per method x dataset (sorted by AUC)")
print(acc1b.to_string())
save(acc1b, "1b_accuracy_method_x_dataset.csv")

# 1b2. AUC per method x dataset
auc1b = (p1.groupby(["method", "dataset"])["auc"]
           .mean()
           .round(4)
           .unstack("dataset"))
auc1b = auc1b.reindex(method_auc_order)
auc1b.index.name = "method"
print("\n[1b2] AUC per method x dataset (sorted by AUC)")
print(auc1b.to_string())
save(auc1b, "1b2_auc_method_x_dataset.csv")

# 1c. DCD variants only — sorted by AUC
dcd = p1[p1["method"].str.startswith("DCD_")]
t1c_mean = (dcd.groupby("method")[["accuracy", "f1", "auc", "time_s", "n_iter"]]
               .mean().round(4))
t1c_std  = (dcd.groupby("method")[["accuracy", "f1", "auc", "time_s", "n_iter"]]
               .std().round(4))
t1c_std.columns = [c + "_std" for c in t1c_std.columns]
t1c = pd.concat([t1c_mean, t1c_std], axis=1)[col_order].sort_values("auc", ascending=False)
print("\n[1c] DCD variants ranking (sorted by AUC)")
print(t1c.to_string())
save(t1c, "1c_dcd_variants_ranking.csv")

# 1d. Stability across seeds (std) — sorted by AUC
t1d_acc = (p1.groupby("method")["accuracy"]
             .agg(["mean", "std", "min", "max"])
             .round(4))
t1d_acc.columns = ["acc_mean", "acc_std", "acc_min", "acc_max"]
t1d_auc = (p1.groupby("method")["auc"]
             .agg(["mean", "std", "min", "max"])
             .round(4))
t1d_auc.columns = ["auc_mean", "auc_std", "auc_min", "auc_max"]
t1d = pd.concat([t1d_acc, t1d_auc], axis=1).sort_values("auc_mean", ascending=False)
print("\n[1d] Stability (accuracy & AUC std over seeds x datasets, sorted by AUC)")
print(t1d.to_string())
save(t1d, "1d_stability.csv")


header("PHASE 2: Impact of penalty parameter C")
p2 = load(2)

# 2a. Accuracy vs C — mean ± std
t2a = (p2.groupby(["method", "C"])["accuracy"]
         .mean()
         .round(4)
         .unstack("C"))
print("\n[2a] Accuracy vs C (mean over datasets x seeds)")
print(t2a.to_string())
save(t2a, "2a_accuracy_vs_C.csv")

t2a_std = (p2.groupby(["method", "C"])["accuracy"]
             .std()
             .round(4)
             .unstack("C"))
t2a_std.columns = [f"std_{c}" for c in t2a_std.columns]
t2a_full = pd.concat([t2a, t2a_std], axis=1)
save(t2a_full, "2a_accuracy_vs_C_with_std.csv")
print("[2a+] With std saved separately")

# 2a2. AUC vs C
t2a2 = (p2.groupby(["method", "C"])["auc"]
          .mean()
          .round(4)
          .unstack("C"))
print("\n[2a2] AUC vs C (mean over datasets x seeds)")
print(t2a2.to_string())
save(t2a2, "2a2_auc_vs_C.csv")

t2a2_std = (p2.groupby(["method", "C"])["auc"]
              .std()
              .round(4)
              .unstack("C"))
t2a2_std.columns = [f"std_{c}" for c in t2a2_std.columns]
t2a2_full = pd.concat([t2a2, t2a2_std], axis=1)
save(t2a2_full, "2a2_auc_vs_C_with_std.csv")

# 2b. Training time vs C — with std
t2b = (p2.groupby(["method", "C"])["time_s"]
         .mean()
         .round(3)
         .unstack("C"))
print("\n[2b] Training time [s] vs C")
print(t2b.to_string())
save(t2b, "2b_time_vs_C.csv")

t2b_std = (p2.groupby(["method", "C"])["time_s"]
             .std()
             .round(3)
             .unstack("C"))
t2b_std.columns = [f"std_{c}" for c in t2b_std.columns]
t2b_full = pd.concat([t2b, t2b_std], axis=1)
save(t2b_full, "2b_time_vs_C_with_std.csv")

# 2c. Accuracy vs C per dataset per method
for method in sorted(p2["method"].unique()):
    sub = p2[p2["method"] == method]
    best = (sub.groupby(["dataset", "C"])["accuracy"]
               .mean()
               .round(4)
               .unstack("C"))
    save(best, f"2c_accuracy_vs_C_{method}.csv")
print("\n[2c] Accuracy vs C per dataset - saved separately per method")

# 2d. Iterations vs C — with std
t2d = (p2.groupby(["method", "C"])["n_iter"]
         .mean()
         .round(1)
         .unstack("C"))
print("\n[2d] Iterations vs C")
print(t2d.to_string())
save(t2d, "2d_iters_vs_C.csv")

t2d_std = (p2.groupby(["method", "C"])["n_iter"]
             .std()
             .round(1)
             .unstack("C"))
t2d_std.columns = [f"std_{c}" for c in t2d_std.columns]
t2d_full = pd.concat([t2d, t2d_std], axis=1)
save(t2d_full, "2d_iters_vs_C_with_std.csv")


header("PHASE 3: Scalability - SUSY subsamples")
p3 = load(3)

# 3a. Accuracy vs fraction — with std
t3a = (p3.groupby(["method", "fraction"])["accuracy"]
         .mean()
         .round(4)
         .unstack("fraction"))
print("\n[3a] Accuracy vs SUSY fraction")
print(t3a.to_string())
save(t3a, "3a_accuracy_vs_fraction.csv")

t3a_std = (p3.groupby(["method", "fraction"])["accuracy"]
             .std()
             .round(4)
             .unstack("fraction"))
t3a_std.columns = [f"std_{c}" for c in t3a_std.columns]
t3a_full = pd.concat([t3a, t3a_std], axis=1)
save(t3a_full, "3a_accuracy_vs_fraction_with_std.csv")

# 3a2. AUC vs fraction
t3a2 = (p3.groupby(["method", "fraction"])["auc"]
          .mean()
          .round(4)
          .unstack("fraction"))
print("\n[3a2] AUC vs SUSY fraction")
print(t3a2.to_string())
save(t3a2, "3a2_auc_vs_fraction.csv")

t3a2_std = (p3.groupby(["method", "fraction"])["auc"]
              .std()
              .round(4)
              .unstack("fraction"))
t3a2_std.columns = [f"std_{c}" for c in t3a2_std.columns]
t3a2_full = pd.concat([t3a2, t3a2_std], axis=1)
save(t3a2_full, "3a2_auc_vs_fraction_with_std.csv")

# 3b. Training time vs fraction — with std
t3b = (p3.groupby(["method", "fraction"])["time_s"]
         .mean()
         .round(2)
         .unstack("fraction"))
print("\n[3b] Training time [s] vs SUSY fraction")
print(t3b.to_string())
save(t3b, "3b_time_vs_fraction.csv")

t3b_std = (p3.groupby(["method", "fraction"])["time_s"]
             .std()
             .round(2)
             .unstack("fraction"))
t3b_std.columns = [f"std_{c}" for c in t3b_std.columns]
t3b_full = pd.concat([t3b, t3b_std], axis=1)
save(t3b_full, "3b_time_vs_fraction_with_std.csv")

# 3c. Full table
t3c = (p3.groupby(["method", "fraction"])[["accuracy", "auc", "time_s", "n_iter"]]
         .agg(["mean", "std"])
         .round(4))
t3c.columns = [f"{m}_{s}" for m, s in t3c.columns]
save(t3c, "3c_full_vs_fraction.csv")
print("\n[3c] Full table (accuracy + auc + time + iter, mean+std) saved")


header("PHASE 4: Impact of sparsity - synthetic data")
p4 = load(4)

# 4a. Accuracy vs sparsity — with std
t4a = (p4.groupby(["method", "sparsity"])["accuracy"]
         .mean()
         .round(4)
         .unstack("sparsity"))
print("\n[4a] Accuracy vs sparsity")
print(t4a.to_string())
save(t4a, "4a_accuracy_vs_sparsity.csv")

t4a_std = (p4.groupby(["method", "sparsity"])["accuracy"]
             .std()
             .round(4)
             .unstack("sparsity"))
t4a_std.columns = [f"std_{c}" for c in t4a_std.columns]
t4a_full = pd.concat([t4a, t4a_std], axis=1)
save(t4a_full, "4a_accuracy_vs_sparsity_with_std.csv")

# 4a2. AUC vs sparsity
t4a2 = (p4.groupby(["method", "sparsity"])["auc"]
          .mean()
          .round(4)
          .unstack("sparsity"))
print("\n[4a2] AUC vs sparsity")
print(t4a2.to_string())
save(t4a2, "4a2_auc_vs_sparsity.csv")

t4a2_std = (p4.groupby(["method", "sparsity"])["auc"]
              .std()
              .round(4)
              .unstack("sparsity"))
t4a2_std.columns = [f"std_{c}" for c in t4a2_std.columns]
t4a2_full = pd.concat([t4a2, t4a2_std], axis=1)
save(t4a2_full, "4a2_auc_vs_sparsity_with_std.csv")

# 4b. Training time vs sparsity — with std
t4b = (p4.groupby(["method", "sparsity"])["time_s"]
         .mean()
         .round(3)
         .unstack("sparsity"))
print("\n[4b] Training time [s] vs sparsity")
print(t4b.to_string())
save(t4b, "4b_time_vs_sparsity.csv")

t4b_std = (p4.groupby(["method", "sparsity"])["time_s"]
             .std()
             .round(3)
             .unstack("sparsity"))
t4b_std.columns = [f"std_{c}" for c in t4b_std.columns]
t4b_full = pd.concat([t4b, t4b_std], axis=1)
save(t4b_full, "4b_time_vs_sparsity_with_std.csv")

# 4c. F1 vs sparsity — with std
t4c = (p4.groupby(["method", "sparsity"])["f1"]
         .mean()
         .round(4)
         .unstack("sparsity"))
print("\n[4c] F1 vs sparsity")
print(t4c.to_string())
save(t4c, "4c_f1_vs_sparsity.csv")

t4c_std = (p4.groupby(["method", "sparsity"])["f1"]
             .std()
             .round(4)
             .unstack("sparsity"))
t4c_std.columns = [f"std_{c}" for c in t4c_std.columns]
t4c_full = pd.concat([t4c, t4c_std], axis=1)
save(t4c_full, "4c_f1_vs_sparsity_with_std.csv")

header("PHASE 5: Convergence analysis - impact of tol")
p5 = load(5)

# 5a. Iterations vs tol — with std
t5a = (p5.groupby(["dataset", "tol"])["n_iter"]
         .mean()
         .round(1)
         .unstack("tol"))
t5a.loc["MEAN"] = t5a.mean().round(1)
print("\n[5a] Iterations vs tol")
print(t5a.to_string())
save(t5a, "5a_iters_vs_tol.csv")

t5a_std = (p5.groupby(["dataset", "tol"])["n_iter"]
             .std()
             .round(1)
             .unstack("tol"))
t5a_std.loc["MEAN"] = t5a_std.mean().round(1)
save(t5a_std, "5a_iters_vs_tol_std.csv")

# 5b. Training time vs tol — with std
t5b = (p5.groupby(["dataset", "tol"])["time_s"]
         .mean()
         .round(3)
         .unstack("tol"))
t5b.loc["MEAN"] = t5b.mean().round(3)
print("\n[5b] Training time [s] vs tol")
print(t5b.to_string())
save(t5b, "5b_time_vs_tol.csv")

t5b_std = (p5.groupby(["dataset", "tol"])["time_s"]
             .std()
             .round(3)
             .unstack("tol"))
t5b_std.loc["MEAN"] = t5b_std.mean().round(3)
save(t5b_std, "5b_time_vs_tol_std.csv")

# 5c. Accuracy vs tol — with std
t5c = (p5.groupby(["dataset", "tol"])["accuracy"]
         .mean()
         .round(4)
         .unstack("tol"))
t5c.loc["MEAN"] = t5c.mean().round(4)
print("\n[5c] Accuracy vs tol")
print(t5c.to_string())
save(t5c, "5c_accuracy_vs_tol.csv")

t5c_std = (p5.groupby(["dataset", "tol"])["accuracy"]
             .std()
             .round(4)
             .unstack("tol"))
t5c_std.loc["MEAN"] = t5c_std.mean().round(4)
save(t5c_std, "5c_accuracy_vs_tol_std.csv")

# 5d. AUC vs tol
t5d = (p5.groupby(["dataset", "tol"])["auc"]
         .mean()
         .round(4)
         .unstack("tol"))
t5d.loc["MEAN"] = t5d.mean().round(4)
print("\n[5d] AUC vs tol")
print(t5d.to_string())
save(t5d, "5d_auc_vs_tol.csv")

t5d_std = (p5.groupby(["dataset", "tol"])["auc"]
             .std()
             .round(4)
             .unstack("tol"))
t5d_std.loc["MEAN"] = t5d_std.mean().round(4)
save(t5d_std, "5d_auc_vs_tol_std.csv")


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