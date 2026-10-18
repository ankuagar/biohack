"""Boil the ImYoo .h5ad file down to small summary tables for the portal.

Run this INSIDE SageMaker Studio (the data must stay in SPC's AWS):
    python precompute.py
It writes a handful of small files to ./data next to this script.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests

H5AD = os.path.expanduser("~/data/imyoo_spc_biohackathon_20261017T211331Z.h5ad")  # v3 (v1 is dummy data)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

JUNK = "Debris|Multiplet|artifact|Unknown|Platelet"
BOWEL = "Crohn|Ulcerative|Ibd|Indeterminate"
LEVELS = {"broad": "cell_type_level_2", "detailed": "cell_type_level_3", "tcell": "cell_type_level_4"}
MIN_CELLS_PER_GROUP = 20  # pseudobulk groups smaller than this are too noisy
N_MAP_CELLS = 60_000      # cells shown on the cell map
N_GENES = 6_000           # genes kept for the gene explorer
# CD8 T-cell inflammation genes, always kept so the IBD vs cancer comparison can show them (the portal groups them)
CD8_PANEL = {
    "Killing": ["GZMB", "GZMA", "GZMH", "GZMK", "PRF1", "GNLY", "NKG7"],
    "Activation / interferon": ["IFNG", "CD69", "HLA-DRA", "CD38", "ISG15", "MX1", "IFI6", "STAT1"],
    "Brakes / checkpoints": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX"],
    "Gut homing": ["ITGA4", "ITGB7", "ITGAE", "CCR9"],
    "Resting / memory": ["CCR7", "SELL", "TCF7", "IL7R"],
}
PANEL_GENES = [g for genes in CD8_PANEL.values() for g in genes]


GENE_TABLE = os.environ.get("IMYOO_GENE_TABLE", os.path.expanduser("~/data/sid/features.tsv.gz"))  # 10x gene table: code <-> name (from Sid's lab output)
KNOWN = ["CD3E", "CD8A", "MS4A1", "CD79A", "NKG7", "GNLY", "CD14", "LYZ", "GZMB", "IL7R", "CCR7", "FCGR3A"]


def readable_gene_names(adata, say=print):
    """Gene names people recognise (GZMB, CD8A...), whichever column of the file holds them.

    v3 lists genes by row number, with names or codes in other columns. Tries every column and picks the one with the
    most well-known gene names; if only codes (ENSG...) exist, translates them with the 10x gene table if available.
    """
    options = {"row names": adata.var_names.astype(str).to_numpy()}
    for c in adata.var.columns:
        if not pd.api.types.is_numeric_dtype(adata.var[c]):  # text columns, whatever pandas calls them
            options[c] = adata.var[c].astype(str).to_numpy()
    score = {k: int(np.isin(KNOWN, v).sum()) for k, v in options.items()}
    best = max(score, key=score.get)
    if score[best] >= 5:
        say(f"gene names from '{best}' ({score[best]} of {len(KNOWN)} well-known genes found)")
        return options[best]
    coded = next((k for k, v in options.items() if np.mean([str(g).startswith("ENS") for g in v[:500]]) > 0.8), None)
    if coded and os.path.exists(GENE_TABLE):
        ft = pd.read_csv(GENE_TABLE, sep="\t", header=None).iloc[:, :2].astype(str)
        name_of = dict(zip(ft[0], ft[1]))
        codes = [str(g).split(".")[0] for g in options[coded]]
        names = np.array([name_of.get(c, c) for c in codes])
        say(f"gene names translated from codes in '{coded}' using the 10x gene table "
            f"({int(np.isin(KNOWN, names).sum())} of {len(KNOWN)} well-known genes found)")
        return names
    say(f"WARNING: no readable gene names found (columns: {list(options)}); using '{coded or 'row names'}'")
    return options[coded or "row names"]


def family_of(cell_type):
    """Group a cell type name into its family (v3 splits e.g. B cells into naive, memory, plasma...)."""
    import re
    n = str(cell_type).lower()
    if "nkt" in n or re.search(r"\bt cells?\b|t follicular|t helper|regulatory t|treg|mait|gamma.?delta", n):
        return "T Cells"
    if re.search(r"\bnk\b|natural killer", n):
        return "NK Cells"
    if "dendritic" in n or re.search(r"\b(c?dc\d?|pdc|asdc)\b", n):  # before B cells: "Plasmacytoid Dendritic"
        return "Dendritic Cells"
    if "plasma" in n or re.search(r"\bb cells?\b", n):
        return "B Cells"
    if "monocyte" in n or "macrophage" in n:
        return "Monocytes"
    if "progenitor" in n or "hspc" in n:
        return "Progenitors"
    if "proliferating" in n or "cycling" in n:
        return "Proliferating Cells"
    return "Other"


def text(col, missing="Unlabelled"):
    """Column as plain strings, with blanks filled in (pandas 3 keeps blanks as NaN after astype(str))."""
    return col.astype(object).where(col.notna(), missing).to_numpy(dtype=str)


def disease_group(d):
    d = str(d)
    if "Collagenous" in d:
        return "Other"
    return "Bowel disease" if re.search(BOWEL, d, flags=re.IGNORECASE) else "Other"


def build_samples(obs):
    """One row per sample, with people renamed P01..P29 (no real IDs on screen)."""
    people = sorted(obs["participant_id"].astype(str).unique())
    alias = {p: f"P{i + 1:02d}" for i, p in enumerate(people)}
    cols = dict(
        person=("participant_id", "first"),
        flare=("flare_status", "first"),
        disease=("canonical_disease", "first"),
        run=("cell_barcoding_run_id", "first"),
        day=("relative_timepoint_seconds", "first"),
        freezer_days=("cell_barcoding_delay_days", "first"),
        hbi=("harvey_bradshaw_index", "first"),
        age=("age_at_start_years", "first"),            # new in v3
        steroid_hours=("hours_since_steroids", "first"),  # new in v3; blank = no recent steroids recorded
        n_cells=("flare_status", "size"),
    )
    s = obs.groupby("sample_id", observed=True).agg(**{k: v for k, v in cols.items() if v[0] in obs.columns}).reset_index()
    s["person"] = s["person"].astype(str).map(alias)
    s["day"] = (s["day"] / 86400).round(1)
    for c in ["flare", "disease", "run"]:
        s[c] = s[c].astype(str)
    s["group"] = s["disease"].map(disease_group)
    return s


def composition(obs, samples):
    """Share (%) of each cell type in each sample, at every detail level. Long format, zeros kept."""
    out = []
    n = samples.set_index("sample_id")["n_cells"]
    for key, col in LEVELS.items():
        if col not in obs.columns:  # e.g. v3 dropped cell_type_level_4
            print(f"   (no {col} in this file; skipping that detail level)")
            continue
        counts = pd.crosstab(obs["sample_id"].astype(str), obs[col].astype(str))
        counts = counts.drop(columns=[c for c in counts.columns if c in ("nan", "None")], errors="ignore")
        pct = counts.div(n.reindex(counts.index), axis=0) * 100
        long = pct.stack().rename("pct").reset_index()
        long.columns = ["sample_id", "cell_type", "pct"]
        long["level"] = key
        out.append(long)
    return pd.concat(out, ignore_index=True)


def gene_tests(pb, samples):
    """Paired flare-vs-baseline test per gene, per cell type, bowel-disease people with both states.

    pb: long table sample_id, cell_type, gene, logcpm.
    """
    s = samples[samples["group"] == "Bowel disease"][["sample_id", "person", "flare"]]
    d = pb.merge(s, on="sample_id")
    rows = []
    for ct, sub in d.groupby("cell_type"):
        per = sub.groupby(["person", "flare", "gene"])["logcpm"].mean().unstack("gene")
        try:
            base = per.xs("Baseline", level="flare")
            flare = per.xs("Flaring", level="flare")
        except KeyError:
            continue
        people = base.index.intersection(flare.index)
        if len(people) < 5:
            continue
        diff = (flare.loc[people] - base.loc[people]).dropna(axis=1)
        n = len(people)
        mean, sd = diff.mean(), diff.std(ddof=1)
        tstat = mean / (sd / np.sqrt(n))
        p = 2 * t_dist.sf(np.abs(tstat.fillna(0)), df=n - 1)
        p = np.where(sd > 0, p, 1.0)
        rows.append(pd.DataFrame({
            "cell_type": ct, "gene": diff.columns,
            "fine": base.loc[people, diff.columns].mean().values,
            "flare": flare.loc[people, diff.columns].mean().values,
            "change": mean.values, "people_up": (diff > 0).sum().values,
            "people": n, "p": p,
        }))
    res = pd.concat(rows, ignore_index=True)
    res["padj"] = res.groupby("cell_type")["p"].transform(lambda x: multipletests(x, method="fdr_bh")[1])
    return res


def main():
    import scanpy as sc

    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=H5AD)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    def say(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    say(f"Loading {args.h5ad} (takes a few minutes)...")
    adata = sc.read_h5ad(args.h5ad)
    obs = adata.obs
    l2 = pd.Series(text(obs["cell_type_level_2"]))
    junk = (l2.str.contains(JUNK + "|Unlabelled", case=False)
            | (text(obs["cell_type_level_1"]) == "Non-WBCs")
            | obs["sample_id"].isna().to_numpy()).to_numpy()
    say(f"{adata.n_obs:,} cells; removing {junk.sum():,} debris/multiplet/artifact/unlabelled cells")
    adata = adata[~junk].copy()
    obs = adata.obs

    say("Samples and cell-type shares...")
    samples = build_samples(obs)
    samples.to_parquet(f"{args.out}/samples.parquet", index=False)
    composition(obs, samples).to_parquet(f"{args.out}/composition.parquet", index=False)

    say("Cell map...")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(adata.n_obs, size=min(N_MAP_CELLS, adata.n_obs), replace=False))
    sub = adata[idx].copy()
    if "X_umap" in sub.obsm:
        xy, map_source = sub.obsm["X_umap"][:, :2], "stored UMAP"
    else:
        rep = next((k for k in sub.obsm if "scvi" in k.lower()), None)
        if rep:
            sc.pp.neighbors(sub, use_rep=rep)
            map_source = f"UMAP of stored {rep}"
        else:
            sc.pp.normalize_total(sub)
            sc.pp.log1p(sub)
            sc.pp.highly_variable_genes(sub, n_top_genes=2000)
            sc.pp.pca(sub, n_comps=30, mask_var="highly_variable")
            sc.pp.neighbors(sub)
            map_source = "UMAP computed from gene activity"
        sc.tl.umap(sub)
        xy = sub.obsm["X_umap"][:, :2]
    person = dict(zip(samples["sample_id"], samples["person"]))
    group = dict(zip(samples["sample_id"], samples["group"]))
    cells = pd.DataFrame({
        "x": xy[:, 0].astype("float32"), "y": xy[:, 1].astype("float32"),
        "broad": text(sub.obs["cell_type_level_2"]),
        "detailed": text(sub.obs["cell_type_level_3"]),
        "flare": text(sub.obs["flare_status"]),
        "person": pd.Series(text(sub.obs["sample_id"])).map(person).values,
        "group": pd.Series(text(sub.obs["sample_id"])).map(group).values,
    })
    cells.to_parquet(f"{args.out}/cells_map.parquet", index=False)
    del sub

    say("Gene activity per sample and cell type (pseudobulk)...")
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = sparse.csr_matrix(X)
    probe = X[:2000].data[:10000]
    if probe.size and not np.allclose(probe, np.round(probe)):
        say("WARNING: gene table doesn't look like raw counts; results are still comparable but check this")
    genes = readable_gene_names(adata, say)
    key = np.char.add(np.char.add(text(obs["sample_id"]), "|"), text(obs["cell_type_level_2"]))
    codes, groups = pd.factorize(key)
    groups = np.asarray(groups)
    G = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(groups), len(codes)))
    pb = np.asarray((G @ X).todense(), dtype="float64")
    n_in_group = np.bincount(codes)
    ok = n_in_group >= MIN_CELLS_PER_GROUP
    pb, groups, n_in_group = pb[ok], groups[ok], n_in_group[ok]
    cpm = pb / pb.sum(axis=1, keepdims=True) * 1e6
    meta = pd.DataFrame([g.split("|", 1) for g in groups], columns=["sample_id", "cell_type"])
    meta["n_cells"] = n_in_group
    # keep genes that are clearly active in at least one cell type, most variable first
    by_type = pd.DataFrame(cpm).groupby(meta["cell_type"].values).mean()
    active = (by_type.max(axis=0) >= 10).values
    logcpm = np.log2(cpm + 1)
    var = logcpm.var(axis=0)
    keep = np.where(active)[0]
    keep = keep[np.argsort(-var[keep])][:N_GENES]
    # always keep the CD8 panel genes, found by name in whichever column of the file holds gene names
    panel_idx = np.where(np.isin(np.asarray(genes), PANEL_GENES))[0]
    if len(panel_idx):
        keep = np.unique(np.concatenate([keep, panel_idx]))
        say(f"kept {len(panel_idx)} CD8 panel genes")
    wide = pd.DataFrame(logcpm[:, keep].astype("float32"), columns=pd.Index(genes[keep]).astype(str))
    wide = wide.loc[:, ~wide.columns.duplicated()]
    pd.concat([meta, wide], axis=1).to_parquet(f"{args.out}/pseudobulk.parquet", index=False)
    say(f"{len(meta):,} sample x cell-type groups, {wide.shape[1]:,} genes kept")

    say("Gene tests (flare vs fine, each person vs themselves)...")
    long = pd.concat([meta[["sample_id", "cell_type"]], wide], axis=1).melt(
        id_vars=["sample_id", "cell_type"], var_name="gene", value_name="logcpm")
    gene_tests(long, samples).to_parquet(f"{args.out}/gene_tests.parquet", index=False)

    info = {
        "demo": False, "built": time.strftime("%Y-%m-%d %H:%M"),
        "cells_total": int(len(junk)), "cells_removed": int(junk.sum()),
        "cells_kept": int(adata.n_obs), "genes_total": int(adata.n_vars),
        "people": int(samples["person"].nunique()), "samples": int(len(samples)),
        "map_source": map_source, "map_cells": int(len(cells)),
    }
    with open(f"{args.out}/info.json", "w") as f:
        json.dump(info, f, indent=2)
    say(f"Done. Files in {args.out}:")
    for name in sorted(os.listdir(args.out)):
        print(f"   {name:24s} {os.path.getsize(os.path.join(args.out, name)) / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
