"""Look inside Sid Sijbrandij's public osteosarcoma files (structure first) to plan the 'Sid's journey' analysis.

Run in SageMaker, after copying the files into ~/data/sid (see README):
    python sid_inspect.py
"""
import os

import pandas as pd

SID = os.path.expanduser("~/data/sid")
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60)

print("=" * 80, "\nCATALOG (what samples exist, and when)")
for name in ["samples.tsv", "specimens.tsv"]:
    path = f"{SID}/catalog/{name}"
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t")
        print(f"\n--- {name}: {len(df)} rows, columns: {list(df.columns)}")
        print(df.head(15).to_string())
    else:
        print(f"(missing {path})")

h5ad = next((f for f in sorted(os.listdir(SID)) if f.endswith(".h5ad")), None)
if h5ad:
    import scanpy as sc

    print("=" * 80, f"\nBLOOD SINGLE-CELL FILE: {h5ad}")
    a = sc.read_h5ad(f"{SID}/{h5ad}", backed="r")
    print(a)
    print("\nLabel columns (name, number of different values, examples):")
    for c in a.obs.columns:
        vals = a.obs[c].dropna().unique()
        print(f"  {c:40s} {len(vals):>8}   e.g. {list(vals[:4])}")
    print("\nMaps stored (obsm):", list(a.obsm.keys()))
    print("Layers:", list(a.layers.keys()))
    print("Gene name examples:", list(a.var_names[:8]))
    x = a.X[:200]
    x = x.toarray() if hasattr(x, "toarray") else x
    print("Looks like raw counts (whole numbers)?", bool((x == x.round()).all()))
else:
    print("(no .h5ad file found in ~/data/sid yet)")
