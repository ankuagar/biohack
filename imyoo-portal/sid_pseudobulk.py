"""Gene activity per blood draw and cell family for Sid Sijbrandij, for the portal's "Compare conditions" page.

Reuses the group labels made by sid_precompute.py (no model re-run). Run INSIDE SageMaker:
    python sid_pseudobulk.py
Writes data/sid/pseudobulk.parquet (+ gene_names.parquet for matching gene names to ImYoo's).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse

from precompute import text
from sid_precompute import FEATURES, SID_H5AD

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sid")
CHUNK = 20_000
MIN_CELLS = 200   # a draw x family group needs this many cells to be trusted


def main():
    import scanpy as sc

    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default=SID_H5AD)
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    def say(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    groups_file = f"{args.out}/groups.parquet"
    if not os.path.exists(groups_file):
        raise SystemExit("Run sid_precompute.py first (it makes the group labels this script reuses).")
    label_of = pd.read_parquet(groups_file).set_index("group")["label"].to_dict()

    sid = sc.read_h5ad(args.sid, backed="r")
    obs = sid.obs
    groups = text(obs["seurat_clusters"]) if "seurat_clusters" in obs else np.full(sid.n_obs, "")
    labels = np.array([label_of.get(g, "Unlabelled") for g in groups])
    dates = text(obs["date"])
    ok = (dates != "Unlabelled") & (dates != "") & (labels != "Unlabelled")
    key = np.where(ok, np.char.add(np.char.add(dates, "|"), labels), "")
    codes, keys = pd.factorize(key)
    keys = np.asarray(keys)
    say(f"{sid.n_obs:,} cells into {len(keys):,} blood-draw x family groups")

    sums = np.zeros((len(keys), sid.n_vars), dtype=np.float64)
    for start in range(0, sid.n_obs, CHUNK):
        stop = min(start + CHUNK, sid.n_obs)
        c = codes[start:stop]
        G = sparse.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(len(keys), len(c)))
        sums += np.asarray((G @ sparse.csr_matrix(sid.X[start:stop])).todense())
        say(f"  {stop:,} / {sid.n_obs:,}")
    n_cells = np.bincount(codes, minlength=len(keys))
    keep = (n_cells >= MIN_CELLS) & (keys != "")
    sums, keys, n_cells = sums[keep], keys[keep], n_cells[keep]
    cpm = sums / sums.sum(axis=1, keepdims=True) * 1e6
    active = cpm.max(axis=0) >= 5   # genes clearly measured somewhere
    genes = sid.var_names.astype(str).to_numpy()[active]
    logcpm = np.log2(cpm[:, active] + 1).astype("float32")

    meta = pd.DataFrame([k.split("|", 1) for k in keys], columns=["date", "cell_type"])
    meta["n_cells"] = n_cells
    wide = pd.DataFrame(logcpm, columns=pd.Index(genes))
    wide = wide.loc[:, ~wide.columns.duplicated()]
    pd.concat([meta, wide], axis=1).to_parquet(f"{args.out}/pseudobulk.parquet", index=False)

    # gene name <-> gene code, so the portal can match Sid's genes with ImYoo's whichever naming ImYoo uses
    names = pd.DataFrame({"name": wide.columns})
    if os.path.exists(args.features):
        ft = pd.read_csv(args.features, sep="\t", header=None).iloc[:, :2].astype(str)
        names["code"] = names["name"].map(dict(zip(ft[1], ft[0])))
    names.to_parquet(f"{args.out}/gene_names.parquet", index=False)
    say(f"Done: {len(meta)} groups, {wide.shape[1]:,} genes -> {args.out}/pseudobulk.parquet")


if __name__ == "__main__":
    main()
