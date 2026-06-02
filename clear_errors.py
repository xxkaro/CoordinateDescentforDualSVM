import os
import pandas as pd

RESULTS_DIR = "results"
HIST_DIR = os.path.join(RESULTS_DIR, "histories")

for fn in os.listdir(RESULTS_DIR):
    if not fn.endswith(".csv"):
        continue
    path = os.path.join(RESULTS_DIR, fn)
    df = pd.read_csv(path)
    if "accuracy" not in df.columns:
        continue
    broken = pd.to_numeric(df["accuracy"], errors="coerce").isna()
    if not broken.any():
        print(f"{fn}: brak błędów")
        continue
    print(f"{fn}: usuwam {broken.sum()} wierszy")
    for _, row in df[broken].iterrows():
        # Usuń odpowiadający JSON
        ds = row.get("dataset", "")
        method = row.get("method", "")
        seed = row.get("seed", "")
        C = row.get("C", "")
        tol = row.get("tol", "")
        hist_fn = f"{ds}_{method}_C{C}_tol{tol}_seed{seed}.json"
        hist_path = os.path.join(HIST_DIR, hist_fn)
        if os.path.exists(hist_path):
            os.remove(hist_path)
            print(f"  usunięto {hist_fn}")
    df[~broken].to_csv(path, index=False)
 