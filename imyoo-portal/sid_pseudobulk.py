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

from precompute import PANEL_GENES, text
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
    # T cells split into CD8 and CD4 from each cell's own marker genes (reliable across lab kits, unlike the model)
    col = {g: i for i, g in enumerate(sid.var_names.astype(str))}
    cd8_cols = [col[g] for g in ["CD8A", "CD8B"] if g in col]
    cd4_cols = [col[g] for g in ["CD4"] if g in col]
    say(f"{sid.n_obs:,} cells; T cells split into CD8/CD4 using {len(cd8_cols)} CD8 and {len(cd4_cols)} CD4 marker genes")

    acc, counts = {}, {}
    def add(keys_chunk, X):
        codes, uniq = pd.factorize(keys_chunk)
        G = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
        S = np.asarray((G @ X).todense())
        for i, k in enumerate(uniq):
            if k:
                acc[k] = acc.get(k, 0) + S[i]
                counts[k] = counts.get(k, 0) + int((codes == i).sum())

    for start in range(0, sid.n_obs, CHUNK):
        stop = min(start + CHUNK, sid.n_obs)
        X = sparse.csr_matrix(sid.X[start:stop])
        d, lab, good = dates[start:stop], labels[start:stop], ok[start:stop]
        add(np.where(good, np.char.add(np.char.add(d, "|"), lab), ""), X)          # whole families
        if cd8_cols and cd4_cols:
            cd8 = np.asarray(X[:, cd8_cols].sum(axis=1)).ravel() > 0
            cd4 = np.asarray(X[:, cd4_cols].sum(axis=1)).ravel() > 0
            is_t = good & (lab == "T Cells")
            sub = np.where(is_t & cd8 & ~cd4, "CD8 T Cells", np.where(is_t & cd4 & ~cd8, "CD4 T Cells", ""))
            add(np.where(sub != "", np.char.add(np.char.add(d, "|"), sub), ""), X)  # CD8 / CD4 subsets
        say(f"  {stop:,} / {sid.n_obs:,}")
    keys = np.array([k for k in acc if counts[k] >= MIN_CELLS])
    sums = np.vstack([acc[k] for k in keys])
    n_cells = np.array([counts[k] for k in keys])
    cpm = sums / sums.sum(axis=1, keepdims=True) * 1e6
    active = (cpm.max(axis=0) >= 5) | np.isin(sid.var_names.astype(str), PANEL_GENES)  # measured, or a CD8 panel gene
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
