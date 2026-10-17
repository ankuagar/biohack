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
T_RECEPTOR_GROUP = 0.4  # a group of cells is T cells if at least this share carry a T-cell receptor
FEW_RECEPTORS = 0.1     # below this share, a group can't be T cells (the model's T/NK mix-ups get corrected)
N_MAP_CELLS = 60_000
N_TOP_CLONES = 30
MARKERS = ["CD3E", "CD8A", "MS4A1", "CD79A", "NKG7", "GNLY", "CD14", "FCGR3A", "LYZ", "FCER1A"]
MARKER_IDS = {"ENSG00000198851", "ENSG00000153563", "ENSG00000156738", "ENSG00000105369", "ENSG00000105374",
              "ENSG00000115523", "ENSG00000170458", "ENSG00000203747", "ENSG00000090382", "ENSG00000179639"}
# 10x's gene table (gene code + gene name) from Sid's lab output, used to translate if the files name genes differently
FEATURES = os.path.expanduser("~/data/sid/features.tsv.gz")


def gene_name_options(adata, translate=None):
    """Every way this file names its genes: the row names, any text column in .var, and (if given) the row
    names translated through 10x's name<->code table.

    Gene ID codes lose their version suffix (ENSG00000156738.18 -> ENSG00000156738) so they match across files.
    """
    opts = {"row names": adata.var_names.astype(str)}
    for c in adata.var.columns:
        if adata.var[c].dtype == object or str(adata.var[c].dtype) == "category":
            opts[c] = adata.var[c].astype(str)
    if translate:
        rows = adata.var_names.astype(str)
        opts["row names via 10x table"] = [translate.get(g, g) for g in rows]
    return {k: np.array([g.split(".")[0] if g.startswith("ENS") else g for g in v]) for k, v in opts.items()}


def match_gene_names(a, b, translate=None):
    """Pick the naming systems (one per file) that share the most genes INCLUDING well-known immune markers."""
    best = (-1, 0, None, None)
    for ka, va in gene_name_options(a, translate).items():
        for kb, vb in gene_name_options(b, translate).items():
            shared = set(va) & set(vb)
            score = (len(shared & (set(MARKERS) | MARKER_IDS)), len(shared))
            if score > best[:2]:
                best = (*score, (ka, va), (kb, vb))
    return best


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
    ap.add_argument("--features", default=FEATURES)
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
    translate = None
    if os.path.exists(args.features):  # name -> code and code -> name
        ft = pd.read_csv(args.features, sep="\t", header=None).iloc[:, :2].astype(str)
        translate = {**dict(zip(ft[1], ft[0])), **dict(zip(ft[0], ft[1]))}
    n_markers, n_shared, (im_key, im_sym), (sid_key, sid_sym) = match_gene_names(train, sid, translate)
    say(f"Gene names matched using ImYoo '{im_key}' and Sid '{sid_key}': {n_shared:,} genes, "
        f"{n_markers} of 10 immune markers")
    if n_markers < 5 or n_shared < min(2000, 0.5 * min(train.n_vars, sid.n_vars)):
        raise SystemExit("The two files name genes too differently to match. Copy Sid's 10x gene table first:\n"
                         "  aws s3 cp s3://sid-sijbrandij-osteosarc-dataset/hudson_lab/PBMC_scRNAseq/cellranger/"
                         "2026-02-13/2026-02-13_capture1/outs/multi/count/raw_feature_bc_matrix/features.tsv.gz "
                         "~/data/sid/ --region us-west-2 --no-sign-request\n"
                         f"(ImYoo gene columns: {list(gene_name_options(train))}; Sid: {list(gene_name_options(sid))})")
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
    model = LogisticRegression(max_iter=1500, C=0.5)
    model.fit(pca.transform(zscore(Xtr, mu_im, sd_im)), y_detail)
    classes = model.classes_
    fam_of_class = np.array([family[c] for c in classes])
    families = sorted(set(fam_of_class))

    # ---- 3. label every Sid cell, a chunk at a time
    say(f"Scoring Sid's {sid.n_obs:,} cells with the model...")
    # markers are gene names; if Sid's file is matched by ID codes, look names up in its other columns
    sym_lookup = sid_col
    if not any(m in sid_col for m in MARKERS):
        for _, names in gene_name_options(sid, translate).items():
            if sum(m in set(names) for m in MARKERS) >= 5:
                sym_lookup = {g: i for i, g in enumerate(names)}
                break
    marker_cols = [sym_lookup[m] for m in MARKERS if m in sym_lookup]
    marker_names = [m for m in MARKERS if m in sym_lookup]
    P_all = np.zeros((sid.n_obs, len(classes)), dtype=np.float32)
    M_all = np.zeros((sid.n_obs, len(marker_cols)), dtype=np.float32)
    for start in range(0, sid.n_obs, CHUNK):
        stop = min(start + CHUNK, sid.n_obs)
        Xs = lognorm(sid.X[start:stop])
        P_all[start:stop] = model.predict_proba(pca.transform(zscore(Xs[:, sid_idx].toarray(), mu_sid, sd_sid)))
        M_all[start:stop] = Xs[:, marker_cols].toarray()
        say(f"  {stop:,} / {sid.n_obs:,}")

    # ---- 3b. label GROUPS of similar cells (Sid's own clusters), using three clues per group:
    #  1) share of cells carrying a T-cell receptor (a hard fact), 2) the model's average vote, 3) marker genes
    say("Labelling groups of similar cells...")
    obs = sid.obs
    fam_p = np.stack([P_all[:, fam_of_class == f].sum(1) for f in families], axis=1)
    fams = np.array(families)
    model_broad = fams[fam_p.argmax(1)]
    tcr_lab = text(obs["TCR_label"]) if "TCR_label" in obs else np.full(sid.n_obs, "none")
    has_tcr = np.isin(tcr_lab, ["alpha_beta", "gamma_delta"])
    groups = text(obs["seurat_clusters"]) if "seurat_clusters" in obs else model_broad
    broad = np.empty(sid.n_obs, dtype=object)
    detail = np.empty(sid.n_obs, dtype=object)
    rows = []
    for g in sorted(np.unique(groups), key=lambda x: (len(x), x)):
        m = groups == g
        t_share = has_tcr[m].mean()
        vote = fam_p[m].mean(0)
        ranked = [fams[i] for i in np.argsort(-vote)]
        if t_share >= T_RECEPTOR_GROUP:
            label, reason = "T Cells", f"{t_share:.0%} carry a T-cell receptor"
        else:
            allowed = [f for f in ranked if not (f == "T Cells" and t_share < FEW_RECEPTORS)]
            label = allowed[0]
            reason = f"model vote {vote[families.index(label)]:.0%}"
            if ranked[0] == "T Cells" and label != "T Cells":
                reason += f"; not T cells (only {t_share:.0%} carry a T-cell receptor)"
        broad[m] = label
        own = fam_of_class == label  # detailed type: best match within the group's family
        detail[m] = classes[own][P_all[m][:, own].argmax(1)] if own.any() else label
        mk = M_all[m].mean(0)
        rows.append({"group": g, "cells": int(m.sum()), "t_receptor_pct": 100 * t_share,
                     "model_top": ranked[0], "model_vote_pct": 100 * vote.max(), "label": label, "reason": reason,
                     "top_markers": ", ".join(marker_names[i] for i in np.argsort(-mk)[:3])})
    groups_table = pd.DataFrame(rows)
    groups_table.to_parquet(f"{args.out}/groups.parquet", index=False)
    print(groups_table.to_string(index=False))
    marker_sum = {f: (M_all[broad == f].sum(0), int((broad == f).sum())) for f in np.unique(broad)}
    model_t_check = float(100 * (model_broad[has_tcr] == "T Cells").mean()) if has_tcr.any() else None
    agreement = float(100 * (model_broad == broad).mean())
    del P_all, M_all

    # ---- 4. summaries for the portal
    raw_dates = pd.Series(text(obs["date"]))
    print("Blood draw dates in the file:", raw_dates.value_counts().sort_index().to_dict())
    date = pd.to_datetime(raw_dates, errors="coerce")
    first = date.min()
    cells = pd.DataFrame({"date": date, "broad": broad, "detailed": detail,
                          "tcr": text(obs["TCR_label"]) if "TCR_label" in obs else "none",
                          "clone": text(obs["clone_TRB"]) if "clone_TRB" in obs else "NA_NA_NA"})
    cells["day"] = (cells["date"] - first).dt.days

    draws = cells.groupby("date").agg(day=("day", "first"), n_cells=("broad", "size")).reset_index()
    if "biological_sample" in obs:
        draws["sample"] = pd.Series(text(obs["biological_sample"])).groupby(cells["date"].values).agg(
            lambda x: ", ".join(sorted(set(x)))).reindex(draws["date"]).to_numpy()
    draws.to_parquet(f"{args.out}/draws.parquet", index=False)

    comp = []
    known = cells.dropna(subset=["date"])
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
        "teaching_cells": int(len(pick)), "genes_used": len(genes), "groups": int(len(groups_table)),
        "tcr_cells": int(len(tcr_cells)),
        "tcr_in_t_groups_pct": float(100 * (tcr_cells["broad"] == "T Cells").mean()) if len(tcr_cells) else None,
        "model_alone_tcr_check_pct": model_t_check, "model_group_agreement_pct": agreement,
        "dates_unreadable": int(date.isna().sum()),
    }
    with open(f"{args.out}/info.json", "w") as f:
        json.dump(info, f, indent=2)
    say("Done. " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()
                             if k in ("cells", "draws", "groups", "tcr_in_t_groups_pct", "model_alone_tcr_check_pct",
                                      "model_group_agreement_pct", "dates_unreadable")))
    for name in sorted(os.listdir(args.out)):
        print(f"   {name:24s} {os.path.getsize(os.path.join(args.out, name)) / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
