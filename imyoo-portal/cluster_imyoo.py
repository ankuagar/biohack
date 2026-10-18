"""Cluster ImYoo single cells and write compact tables for the Streamlit portal.

Run in SageMaker after installing scanpy, leidenalg and igraph:

    pip install -q scanpy leidenalg igraph pyarrow
    python cluster_imyoo.py

The script uses a stored scVI representation when one is present. Otherwise it
uses a stored PCA representation, or computes a small PCA from normalized,
highly-variable genes. It removes the same junk labels as precompute.py and
never writes the expression matrix to the portal output.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


JUNK = "Debris|Multiplet|artifact|Unknown|Unlabelled|Platelet"


def text(values):
    return pd.Series(values, dtype="string").fillna("Unknown").astype(str)


def choose_rep(adata):
    keys = list(adata.obsm.keys())
    scvi = [k for k in keys if "scvi" in k.lower()]
    if scvi:
        return scvi[0]
    if "X_pca" in adata.obsm:
        return "X_pca"
    return None


def clean_cells(adata):
    obs = adata.obs
    level2 = text(obs["cell_type_level_2"]) if "cell_type_level_2" in obs else pd.Series("Unknown", index=obs.index)
    level1 = text(obs["cell_type_level_1"]) if "cell_type_level_1" in obs else pd.Series("", index=obs.index)
    keep = ~level2.str.contains(JUNK, case=False, regex=True)
    keep &= ~level1.str.fullmatch("Non-WBCs", case=False)
    if "sample_id" in obs:
        keep &= obs["sample_id"].notna()
    return adata[keep.to_numpy()].copy()


def add_sample_labels(adata):
    obs = adata.obs
    def get(name, default):
        return text(obs[name]) if name in obs else pd.Series(default, index=obs.index)
    adata.obs["__sample"] = get("sample_id", "Unknown")
    adata.obs["__person"] = get("participant_id", get("person", "Unknown"))
    adata.obs["__flare"] = get("flare_status", get("flare", "Unknown"))
    adata.obs["__broad"] = get("cell_type_level_1", "Unknown")
    adata.obs["__detailed"] = get("cell_type_level_2", "Unknown")


def prepare_rep(adata, sc):
    rep = choose_rep(adata)
    if rep is not None:
        return adata, rep, False
    # Keep this fallback modest so it works on a 64 GB SageMaker instance.
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger")
    if adata.var["highly_variable"].sum() > 50:
        adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, svd_solver="arpack")
    return adata, "X_pca", True


def write_cell_table(adata, out_dir, max_cells):
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = np.arange(n) if n <= max_cells else np.sort(rng.choice(n, max_cells, replace=False))
    ob = adata.obs.iloc[idx]
    xy = np.asarray(adata.obsm["X_umap"])[idx]
    out = pd.DataFrame({
        "x": xy[:, 0].astype("float32"),
        "y": xy[:, 1].astype("float32"),
        "cluster": text(ob["cluster"]).to_numpy(),
        "broad": text(ob["__broad"]).to_numpy(),
        "detailed": text(ob["__detailed"]).to_numpy(),
        "flare": text(ob["__flare"]).to_numpy(),
        "person": text(ob["__person"]).to_numpy(),
        "sample_id": text(ob["__sample"]).to_numpy(),
    })
    out.to_parquet(os.path.join(out_dir, "cluster_cells.parquet"), index=False)


def summary_tables(adata, out_dir):
    obs = adata.obs
    meta = pd.DataFrame({
        "sample_id": text(obs["__sample"]),
        "person": text(obs["__person"]),
        "flare": text(obs["__flare"]),
        "cluster": text(obs["cluster"]),
        "broad": text(obs["__broad"]),
        "detailed": text(obs["__detailed"]),
    })
    counts = meta.groupby("cluster").size().rename("n_cells")
    summary = counts.to_frame()
    summary["people"] = meta.groupby("cluster")["person"].nunique()
    summary["samples"] = meta.groupby("cluster")["sample_id"].nunique()
    summary["flare_cells"] = meta.assign(is_flare=meta["flare"].str.contains("flare", case=False)) \
        .groupby("cluster")["is_flare"].sum()
    summary["flare_pct"] = summary["flare_cells"] / summary["n_cells"] * 100
    summary["dominant_broad"] = meta.groupby("cluster")["broad"].agg(lambda s: s.value_counts().index[0])
    summary["dominant_detailed"] = meta.groupby("cluster")["detailed"].agg(lambda s: s.value_counts().index[0])
    sample_cluster = pd.crosstab([meta["person"], meta["sample_id"]], meta["cluster"], normalize="index") * 100
    sample_meta = meta.drop_duplicates("sample_id").set_index(["person", "sample_id"])["flare"]
    sample_cluster = sample_cluster.join(sample_meta.rename("flare"))
    deltas = []
    for cluster in sorted(meta["cluster"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        if cluster not in sample_cluster:
            continue
        wide = sample_cluster[[cluster, "flare"]].rename(columns={cluster: "pct"}).reset_index()
        wide = wide[wide["flare"].isin(["Baseline", "Flaring", "Feeling fine", "Flare"])]
        wide["state"] = np.where(wide["flare"].str.contains("flare", case=False), "Flaring", "Baseline")
        per_person = wide.pivot_table(index="person", columns="state", values="pct", aggfunc="mean")
        if {"Baseline", "Flaring"}.issubset(per_person.columns):
            delta = per_person["Flaring"] - per_person["Baseline"]
            deltas.append({"cluster": cluster, "people_with_both": int(delta.notna().sum()),
                           "mean_delta_pct_points": float(delta.mean()),
                           "people_up": int((delta > 0).sum())})
    if deltas:
        summary = summary.reset_index().merge(pd.DataFrame(deltas), on="cluster", how="left").set_index("cluster")
    else:
        summary["people_with_both"] = 0
        summary["mean_delta_pct_points"] = np.nan
        summary["people_up"] = 0
    summary.reset_index().to_parquet(os.path.join(out_dir, "cluster_summary.parquet"), index=False)


def marker_table(adata, out_dir, sc, markers, max_cells):
    rng = np.random.default_rng(1)
    idx = np.arange(adata.n_obs) if adata.n_obs <= max_cells else np.sort(rng.choice(adata.n_obs, max_cells, replace=False))
    m = adata[idx].copy()
    if "counts" in m.layers:
        m.X = m.layers["counts"].copy()
        sc.pp.normalize_total(m, target_sum=1e4)
        sc.pp.log1p(m)
    try:
        sc.tl.rank_genes_groups(m, groupby="cluster", method="wilcoxon", n_genes=markers,
                                use_raw=False, key_added="cluster_markers")
        rows = []
        result = m.uns["cluster_markers"]
        for cluster in result["names"].dtype.names:
            for rank, gene in enumerate(result["names"][cluster]):
                rows.append({"cluster": str(cluster), "rank": rank + 1, "gene": str(gene),
                             "score": float(result["scores"][cluster][rank]),
                             "p_adj": float(result["pvals_adj"][cluster][rank])})
        pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "cluster_markers.parquet"), index=False)
    except Exception as exc:
        print(f"Marker ranking skipped: {exc}")
        pd.DataFrame(columns=["cluster", "rank", "gene", "score", "p_adj"]).to_parquet(
            os.path.join(out_dir, "cluster_markers.parquet"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.expanduser("~/data/imyoo_spc_biohackathon_20261017T211331Z.h5ad"))
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--resolution", type=float, default=0.6)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--markers", type=int, default=10)
    ap.add_argument("--max-cells", type=int, default=100000)
    args = ap.parse_args()
    import scanpy as sc
    sc.settings.verbosity = 1
    print(f"Reading {args.input}")
    adata = clean_cells(sc.read_h5ad(args.input))
    add_sample_labels(adata)
    adata, rep, computed = prepare_rep(adata, sc)
    sc.pp.neighbors(adata, n_neighbors=args.neighbors, use_rep=rep)
    try:
        sc.tl.leiden(adata, resolution=args.resolution, key_added="cluster", flavor="igraph",
                     directed=False, n_iterations=2)
    except TypeError:
        sc.tl.leiden(adata, resolution=args.resolution, key_added="cluster")
    sc.tl.umap(adata, random_state=0)
    os.makedirs(args.out_dir, exist_ok=True)
    write_cell_table(adata, args.out_dir, args.max_cells)
    summary_tables(adata, args.out_dir)
    marker_table(adata, args.out_dir, sc, args.markers, args.max_cells)
    info = {"source": os.path.abspath(args.input), "representation": rep, "representation_computed": computed,
            "resolution": args.resolution, "neighbors": args.neighbors, "n_cells": int(adata.n_obs),
            "n_clusters": int(adata.obs["cluster"].nunique()), "max_cells_in_map": args.max_cells}
    with open(os.path.join(args.out_dir, "cluster_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
