"""Sid Sijbrandij's blood over time: label his immune cells using ImYoo's expert-labelled atlas, then
boil everything down to small tables for the portal's "Sid's journey" page.

Run INSIDE SageMaker after copying the files (see README):
    python sid_precompute.py
Writes small files to ./data/sid next to this script.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse

from precompute import JUNK, text

SID_H5AD = os.path.expanduser("~/data/sid/combined_seurat_object_20260323.h5ad")
IMYOO_H5AD = os.path.expanduser("~/data/imyoo_spc_biohackathon_20261016T004303Z.h5ad")
CATALOG = os.path.expanduser("~/data/sid/catalog/samples.tsv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sid")

PER_TYPE = 4000        # ImYoo training cells per detailed cell type
N_GENES = 3000         # most variable genes shared by both datasets
N_PCS = 40
CHUNK = 20_000         # Sid cells labelled at a time (keeps memory low)
MIN_CONFIDENCE = 0.6   # below this, a cell is called "Uncertain"
N_MAP_CELLS = 60_000
N_TOP_CLONES = 30
MARKERS = ["CD3E", "CD8A", "MS4A1", "CD79A", "NKG7", "GNLY", "CD14", "FCGR3A", "LYZ", "FCER1A"]


def symbols(adata):
    for c in ["gene_symbols", "gene_name", "feature_name", "symbol"]:
        if c in adata.var.columns:
            return adata.var[c].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()


def lognorm(X):
    """Counts -> log(1 + counts per 10,000), the usual single-cell scaling."""
    X = sparse.csr_matrix(X, dtype=np.float32)
    total = np.asarray(X.sum(axis=1)).ravel()
    total[total == 0] = 1
    X = sparse.diags(1e4 / total) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def main():
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default=SID_H5AD)
    ap.add_argument("--imyoo", default=IMYOO_H5AD)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)

    def say(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    # ---- 1. teaching set: a balanced sample of ImYoo's expert-labelled cells
    say("Reading ImYoo's labelled atlas...")
    im = sc.read_h5ad(args.imyoo, backed="r")
    l2, l3 = text(im.obs["cell_type_level_2"]), text(im.obs["cell_type_level_3"])
    good = ~(pd.Series(l2).str.contains(JUNK + "|Unlabelled", case=False)
             | pd.Series(l3).str.contains(JUNK + "|Unlabelled", case=False)).to_numpy()
    good &= text(im.obs["cell_type_level_1"]) != "Non-WBCs"
    pick = []
    for t in np.unique(l3[good]):
        idx = np.where(good & (l3 == t))[0]
        pick.append(rng.choice(idx, size=min(PER_TYPE, len(idx)), replace=False))
    pick = np.sort(np.concatenate(pick))
    train = im[pick].to_memory()
    y_detail = l3[pick]
    family = pd.Series(l2[pick]).groupby(y_detail).agg(lambda x: x.value_counts().index[0]).to_dict()
    say(f"Teaching set: {len(pick):,} cells, {len(family)} cell types")

    # ---- 2. genes both datasets measured; keep the most variable ones
    sid = sc.read_h5ad(args.sid, backed="r")
    im_sym, sid_sym = symbols(train), symbols(sid)
    shared = sorted(set(im_sym) & set(sid_sym))
    Xtr_all = lognorm(train.X)
    im_col = {g: i for i, g in enumerate(im_sym)}
    cols = np.array([im_col[g] for g in shared])
    sub = Xtr_all[:, cols]
    mean = np.asarray(sub.mean(axis=0)).ravel()
    var = np.asarray(sub.multiply(sub).mean(axis=0)).ravel() - mean ** 2
    keep = np.argsort(-(var * (mean > 0.01)))[:N_GENES]
    genes = [shared[i] for i in keep]
    say(f"{len(shared):,} genes shared; using the {len(genes):,} most variable")
    im_idx = np.array([im_col[g] for g in genes])
    sid_col = {g: i for i, g in enumerate(sid_sym)}
    sid_idx = np.array([sid_col[g] for g in genes])

    def zscore(X, mu, sd):
        return np.clip((X - mu) / sd, -10, 10)

    # Each dataset is scaled by its own averages, which evens out differences between the two labs' kits.
    Xtr = Xtr_all[:, im_idx].toarray()
    mu_im, sd_im = Xtr.mean(0), Xtr.std(0) + 1e-3
    probe = np.sort(rng.choice(sid.n_obs, size=min(30_000, sid.n_obs), replace=False))
    Xp = lognorm(sid.X[probe])[:, sid_idx].toarray()
    mu_sid, sd_sid = Xp.mean(0), Xp.std(0) + 1e-3

    say("Teaching the model...")
    pca = PCA(n_components=N_PCS, random_state=0).fit(zscore(Xtr, mu_im, sd_im))
    model = LogisticRegression(max_iter=600, C=0.5)
    model.fit(pca.transform(zscore(Xtr, mu_im, sd_im)), y_detail)
    classes = model.classes_
    fam_of_class = np.array([family[c] for c in classes])
    families = sorted(set(fam_of_class))

    # ---- 3. label every Sid cell, a chunk at a time
    say(f"Labelling Sid's {sid.n_obs:,} cells...")
    detail = np.empty(sid.n_obs, dtype=object)
    broad = np.empty(sid.n_obs, dtype=object)
    conf = np.zeros(sid.n_obs, dtype=np.float32)
    marker_cols = [sid_col[m] for m in MARKERS if m in sid_col]
    marker_names = [m for m in MARKERS if m in sid_col]
    marker_sum = {}
    for start in range(0, sid.n_obs, CHUNK):
        stop = min(start + CHUNK, sid.n_obs)
        Xs = lognorm(sid.X[start:stop])
        P = model.predict_proba(pca.transform(zscore(Xs[:, sid_idx].toarray(), mu_sid, sd_sid)))
        fam_p = np.stack([P[:, fam_of_class == f].sum(1) for f in families], axis=1)
        best_f = fam_p.argmax(1)
        conf[start:stop] = fam_p[np.arange(len(P)), best_f]
        broad[start:stop] = np.array(families)[best_f]
        detail[start:stop] = classes[P.argmax(1)]
        M = Xs[:, marker_cols].toarray()
        for f in families:  # running marker averages per predicted family, for the "are labels right?" check
            rows = broad[start:stop] == f
            if rows.any():
                s, n = marker_sum.get(f, (np.zeros(len(marker_cols)), 0))
                marker_sum[f] = (s + M[rows].sum(0), n + rows.sum())
        say(f"  {stop:,} / {sid.n_obs:,}")
    uncertain = conf < MIN_CONFIDENCE
    broad[uncertain] = "Uncertain"
    detail[uncertain] = "Uncertain"

    # ---- 4. summaries for the portal
    obs = sid.obs
    date = pd.to_datetime(text(obs["date"]), errors="coerce")
    first = date.min()
    cells = pd.DataFrame({"date": date, "broad": broad, "detailed": detail, "confidence": conf,
                          "tcr": text(obs["TCR_label"]) if "TCR_label" in obs else "none",
                          "clone": text(obs["clone_TRB"]) if "clone_TRB" in obs else "NA_NA_NA"})
    cells["day"] = (cells["date"] - first).dt.days

    draws = cells.groupby("date").agg(day=("day", "first"), n_cells=("broad", "size"),
                                      uncertain_pct=("broad", lambda b: 100 * (b == "Uncertain").mean())).reset_index()
    if "biological_sample" in obs:
        draws["sample"] = pd.Series(text(obs["biological_sample"])).groupby(cells["date"].values).agg(
            lambda x: ", ".join(sorted(set(x)))).reindex(draws["date"]).to_numpy()
    draws.to_parquet(f"{args.out}/draws.parquet", index=False)

    comp = []
    known = cells[cells["broad"] != "Uncertain"]
    for level in ["broad", "detailed"]:
        pct = pd.crosstab(known["date"], known[level], normalize="index").mul(100).stack().rename("pct").reset_index()
        pct.columns = ["date", "cell_type", "pct"]
        pct["level"] = level
        comp.append(pct)
    pd.concat(comp).to_parquet(f"{args.out}/composition.parquet", index=False)

    take = np.sort(rng.choice(sid.n_obs, size=min(N_MAP_CELLS, sid.n_obs), replace=False))
    xy = np.asarray(sid.obsm["X_umap"][take])[:, :2] if "X_umap" in sid.obsm else np.zeros((len(take), 2))
    cmap = cells.iloc[take][["date", "broad", "detailed", "tcr"]].copy()
    cmap["x"], cmap["y"] = xy[:, 0].astype("float32"), xy[:, 1].astype("float32")
    cmap.to_parquet(f"{args.out}/cells_map.parquet", index=False)

    t = cells[(cells["tcr"] == "alpha_beta") & ~cells["clone"].str.startswith("NA_") & (cells["clone"] != "")]
    per_date = t.groupby("date")["clone"]
    div = []
    for d, cl in per_date:
        f = cl.value_counts(normalize=True)
        h = -(f * np.log(f)).sum()
        div.append({"date": d, "t_cells_with_receptor": len(cl), "families": len(f),
                    "top10_share": 100 * f.head(10).sum(),
                    "effective_families": float(np.exp(h))})  # drops when a few families take over
    pd.DataFrame(div).to_parquet(f"{args.out}/tcell_diversity.parquet", index=False)
    freq = t.groupby(["date", "clone"]).size().div(per_date.size(), level="date").mul(100).rename("pct").reset_index()
    top = freq.groupby("clone")["pct"].max().sort_values(ascending=False).head(N_TOP_CLONES).index
    clones = freq[freq["clone"].isin(top)].pivot(index="date", columns="clone", values="pct").fillna(0)
    clones = clones.stack().rename("pct").reset_index()
    rank = {c: f"Family {i + 1}" for i, c in enumerate(top)}
    clones["family"] = clones["clone"].map(rank)
    clones.to_parquet(f"{args.out}/clones.parquet", index=False)

    tcr_cells = cells[cells["tcr"].isin(["alpha_beta", "gamma_delta"])]
    labelled_tcr = tcr_cells[tcr_cells["broad"] != "Uncertain"]
    markers = pd.DataFrame({f: s / max(n, 1) for f, (s, n) in marker_sum.items()}, index=marker_names).T
    markers.reset_index(names="predicted family").to_parquet(f"{args.out}/markers.parquet", index=False)

    events = []
    if os.path.exists(args.catalog):
        cat = pd.read_csv(args.catalog, sep="\t")
        tum = cat[cat["tissue"].astype(str).str.lower() == "tumor"].dropna(subset=["collection_date"])
        for _, r in tum.drop_duplicates("collection_date").iterrows():
            events.append({"date": r["collection_date"], "label": str(r["display_name"])})
    pd.DataFrame(events or [{"date": None, "label": None}]).to_parquet(f"{args.out}/events.parquet", index=False)

    info = {
        "built": time.strftime("%Y-%m-%d %H:%M"), "cells": int(sid.n_obs), "draws": int(draws["date"].nunique()),
        "first_draw": str(first.date()), "last_draw": str(date.max().date()),
        "teaching_cells": int(len(pick)), "genes_used": len(genes), "min_confidence": MIN_CONFIDENCE,
        "uncertain_pct": float(100 * uncertain.mean()),
        "tcr_cells": int(len(tcr_cells)),
        "tcr_called_t_pct": float(100 * (labelled_tcr["broad"] == "T Cells").mean()) if len(labelled_tcr) else None,
    }
    with open(f"{args.out}/info.json", "w") as f:
        json.dump(info, f, indent=2)
    say("Done. " + ", ".join(f"{k}={v}" for k, v in info.items() if k in ("cells", "draws", "uncertain_pct",
                                                                         "tcr_called_t_pct")))
    for name in sorted(os.listdir(args.out)):
        print(f"   {name:24s} {os.path.getsize(os.path.join(args.out, name)) / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
