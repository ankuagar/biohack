"""Run a reproducible single-cell clustering parameter study.

The study reuses the same cleaned cells and expression representation for a grid
of Leiden resolutions and neighbor counts. It writes compact tables that can be
reviewed in the Streamlit portal without copying the expression matrix.
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

from cluster_imyoo import add_sample_labels, clean_cells, prepare_rep, text


def parse_numbers(value, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def config_key(neighbors, resolution):
    res = f"{resolution:g}".replace(".", "p")
    return f"cluster_n{neighbors}_r{res}"


def make_meta(adata):
    obs = adata.obs
    return pd.DataFrame({
        "sample_id": text(obs["__sample"]).to_numpy(),
        "person": text(obs["__person"]).to_numpy(),
        "flare": text(obs["__flare"]).to_numpy(),
        "broad": text(obs["__broad"]).to_numpy(),
        "detailed": text(obs["__detailed"]).to_numpy(),
    })


def paired_flare_delta(meta):
    sample_cluster = pd.crosstab(
        [meta["person"], meta["sample_id"]], meta["cluster"], normalize="index"
    ) * 100
    sample_meta = meta.drop_duplicates("sample_id").set_index(["person", "sample_id"])["flare"]
    sample_cluster = sample_cluster.join(sample_meta.rename("flare"))
    wide = sample_cluster.reset_index()
    wide = wide[wide["flare"].isin(["Baseline", "Flaring", "Feeling fine", "Flare"])]
    if wide.empty:
        return {"people_with_both": 0, "mean_delta_pct_points": np.nan, "people_up": 0}
    wide["state"] = np.where(wide["flare"].str.contains("flare", case=False), "Flaring", "Baseline")
    rows = []
    cluster_cols = [c for c in sample_cluster.columns if c != "flare"]
    for cluster in cluster_cols:
        per_person = wide.pivot_table(index="person", columns="state", values=cluster, aggfunc="mean")
        if {"Baseline", "Flaring"}.issubset(per_person.columns):
            delta = per_person["Flaring"] - per_person["Baseline"]
            rows.extend(delta.dropna().tolist())
    if not rows:
        return {"people_with_both": 0, "mean_delta_pct_points": np.nan, "people_up": 0}
    # This aggregate is a study-level diagnostic; cluster-specific deltas are
    # written by summarize_config below.
    return {"people_with_both": len(rows), "mean_delta_pct_points": float(np.mean(rows)),
            "people_up": int(sum(value > 0 for value in rows))}


def summarize_config(meta, cluster_col, config, out):
    frame = meta.copy()
    frame["cluster"] = text(frame[cluster_col])
    counts = frame.groupby("cluster").size().rename("n_cells").to_frame()
    counts["pct_cells"] = counts["n_cells"] / len(frame) * 100
    counts["people"] = frame.groupby("cluster")["person"].nunique()
    counts["samples"] = frame.groupby("cluster")["sample_id"].nunique()
    counts["flare_cells"] = frame.assign(is_flare=frame["flare"].str.contains("flare", case=False)) \
        .groupby("cluster")["is_flare"].sum()
    counts["flare_pct"] = counts["flare_cells"] / counts["n_cells"] * 100
    counts["dominant_broad"] = frame.groupby("cluster")["broad"].agg(lambda s: s.value_counts().index[0])
    counts["dominant_detailed"] = frame.groupby("cluster")["detailed"].agg(lambda s: s.value_counts().index[0])

    delta_rows = []
    for cluster in counts.index:
        sample_cluster = pd.crosstab([frame["person"], frame["sample_id"]], frame["cluster"], normalize="index") * 100
        sample_meta = frame.drop_duplicates("sample_id").set_index(["person", "sample_id"])["flare"]
        sample_cluster = sample_cluster.join(sample_meta.rename("flare"))
        if cluster not in sample_cluster:
            continue
        wide = sample_cluster[[cluster, "flare"]].rename(columns={cluster: "pct"}).reset_index()
        wide = wide[wide["flare"].isin(["Baseline", "Flaring", "Feeling fine", "Flare"])]
        wide["state"] = np.where(wide["flare"].str.contains("flare", case=False), "Flaring", "Baseline")
        per_person = wide.pivot_table(index="person", columns="state", values="pct", aggfunc="mean")
        if {"Baseline", "Flaring"}.issubset(per_person.columns):
            delta = (per_person["Flaring"] - per_person["Baseline"]).dropna()
            delta_rows.append({
                "cluster": cluster,
                "people_with_both": int(delta.size),
                "mean_delta_pct_points": float(delta.mean()) if len(delta) else np.nan,
                "people_up": int((delta > 0).sum()),
            })
    deltas = pd.DataFrame(delta_rows).set_index("cluster") if delta_rows else pd.DataFrame()
    if not deltas.empty:
        counts = counts.join(deltas)
    else:
        counts["people_with_both"] = 0
        counts["mean_delta_pct_points"] = np.nan
        counts["people_up"] = 0
    counts = counts.reset_index()
    counts.insert(0, "config", config["label"])
    counts.insert(1, "config_key", config["key"])
    counts.insert(2, "neighbors", config["neighbors"])
    counts.insert(3, "resolution", config["resolution"])
    out.append(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--neighbors-list", default="15")
    ap.add_argument("--max-cells", type=int, default=100000)
    args = ap.parse_args()
    import scanpy as sc

    resolutions = parse_numbers(args.resolutions, float)
    neighbors_list = parse_numbers(args.neighbors_list, int)
    if not resolutions or not neighbors_list:
        raise ValueError("At least one resolution and one neighbor count are required")

    sc.settings.verbosity = 1
    print(f"Reading {args.input}")
    adata = clean_cells(sc.read_h5ad(args.input))
    add_sample_labels(adata)
    adata, rep, computed = prepare_rep(adata, sc)
    meta = make_meta(adata)
    configs = []
    summary_parts = []
    assignments = {}
    neighbor_keys = {}

    for neighbors in neighbors_list:
        neighbor_key = f"study_neighbors_n{neighbors}"
        sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep=rep, key_added=neighbor_key)
        neighbor_keys[neighbors] = neighbor_key
        for resolution in resolutions:
            key = config_key(neighbors, resolution)
            label = f"{neighbors} neighbors · resolution {resolution:g}"
            config = {"key": key, "label": label, "neighbors": neighbors, "resolution": resolution}
            try:
                sc.tl.leiden(adata, resolution=resolution, key_added=key, neighbors_key=neighbor_key,
                             flavor="igraph", directed=False, n_iterations=2)
            except TypeError:
                sc.tl.leiden(adata, resolution=resolution, key_added=key, neighbors_key=neighbor_key)
            assignments[key] = text(adata.obs[key]).to_numpy()
            configs.append(config)
            summarize_config(pd.concat([meta, pd.Series(assignments[key], name=key)], axis=1), key, config,
                             summary_parts)

    # Use the 15-neighbor, 0.6-resolution map when available; otherwise use the first graph.
    base_neighbors = 15 if 15 in neighbor_keys else neighbors_list[0]
    base_key = config_key(base_neighbors, 0.6) if 0.6 in resolutions else configs[0]["key"]
    sc.tl.umap(adata, neighbors_key=neighbor_keys[base_neighbors], random_state=0)
    rng = np.random.default_rng(0)
    idx = np.arange(adata.n_obs) if adata.n_obs <= args.max_cells else np.sort(rng.choice(adata.n_obs, args.max_cells, replace=False))
    study_cells = meta.iloc[idx].copy()
    xy = np.asarray(adata.obsm["X_umap"])[idx]
    study_cells.insert(0, "x", xy[:, 0].astype("float32"))
    study_cells.insert(1, "y", xy[:, 1].astype("float32"))
    for key, values in assignments.items():
        study_cells[key] = values[idx]

    stability_rows = []
    try:
        from sklearn.metrics import adjusted_rand_score
        for left, right in combinations(configs, 2):
            a = assignments[left["key"]][idx]
            b = assignments[right["key"]][idx]
            stability_rows.append({
                "config_a": left["label"], "config_b": right["label"],
                "adjusted_rand_index": float(adjusted_rand_score(a, b)),
                "same_cluster_fraction": float(np.mean(a == b)),
            })
    except Exception as exc:
        print(f"Stability metrics skipped: {exc}")

    os.makedirs(args.out_dir, exist_ok=True)
    study_cells.to_parquet(os.path.join(args.out_dir, "cluster_study_cells.parquet"), index=False)
    pd.concat(summary_parts, ignore_index=True).to_parquet(
        os.path.join(args.out_dir, "cluster_study_summary.parquet"), index=False
    )
    pd.DataFrame(stability_rows).to_parquet(
        os.path.join(args.out_dir, "cluster_study_stability.parquet"), index=False
    )
    info = {
        "source": os.path.abspath(args.input), "representation": rep,
        "representation_computed": computed, "n_cells": int(adata.n_obs),
        "max_cells_in_map": int(args.max_cells), "base_config": base_key,
        "configs": configs,
    }
    with open(os.path.join(args.out_dir, "cluster_study_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
