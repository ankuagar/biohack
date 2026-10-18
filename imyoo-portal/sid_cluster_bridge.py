"""Score Sid's blood-draw summaries against ImYoo cluster marker programs.

This is a cross-dataset signature comparison. It does not transfer ImYoo
cluster labels to Sid's cells. Scores are calculated from shared genes in the
Sid pseudobulk tables, which avoids pretending that the two lab kits measure
the same raw expression scale.

Run in SageMaker after sid_pseudobulk.py and the ImYoo clustering outputs:

    python sid_cluster_bridge.py

Outputs are compact and written to data/:
    sid_cluster_scores.parquet
    sid_cluster_null_scores.parquet
    sid_cluster_bridge_summary.parquet
    sid_cluster_bridge_info.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def norm_gene(value):
    value = str(value).strip()
    return value.split(".", 1)[0].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markers", default="data/cluster_markers.parquet")
    ap.add_argument("--sid-pseudobulk", default="data/sid/pseudobulk.parquet")
    ap.add_argument("--sid-gene-names", default="data/sid/gene_names.parquet")
    ap.add_argument("--imyoo-h5ad", default=os.path.expanduser(
        "~/data/imyoo_spc_biohackathon_20261017T211331Z.h5ad"),
        help="ImYoo H5AD used to translate numeric cluster marker IDs to gene symbols")
    ap.add_argument("--out", default="data")
    ap.add_argument("--top-markers", type=int, default=10)
    ap.add_argument("--null-sets", type=int, default=100,
                    help="Matched-size random gene sets per cluster for null controls")
    args = ap.parse_args()

    markers = pd.read_parquet(args.markers)
    sid = pd.read_parquet(args.sid_pseudobulk)
    metadata = {"date", "cell_type", "n_cells"}
    gene_columns = [c for c in sid.columns if c not in metadata]
    if not gene_columns:
        raise SystemExit("Sid pseudobulk has no gene columns")

    # Build aliases for both symbols and stable gene IDs. sid_pseudobulk writes
    # the original expression-column name in gene_names.name and the optional
    # 10x code in gene_names.code.
    alias_to_column = {norm_gene(c): c for c in gene_columns}
    if os.path.exists(args.sid_gene_names):
        names = pd.read_parquet(args.sid_gene_names)
        for _, row in names.iterrows():
            column = row.get("name")
            if column in sid.columns:
                for field in ("name", "code"):
                    value = row.get(field)
                    if pd.notna(value):
                        alias_to_column[norm_gene(value)] = column

    # The HealthOmics marker table may contain the H5AD's numeric var_names
    # (for example, "9832") rather than symbols. Translate those IDs using
    # the small .var metadata read from the original H5AD in backed mode.
    marker_id_to_symbol = {}
    mapping_source = None
    if os.path.exists(args.imyoo_h5ad):
        try:
            import scanpy as sc
            imyoo = sc.read_h5ad(args.imyoo_h5ad, backed="r")
            row_names = np.asarray(imyoo.var_names.astype(str))
            options = {"row names": row_names}
            for column in imyoo.var.columns:
                if not pd.api.types.is_numeric_dtype(imyoo.var[column]):
                    options[column] = imyoo.var[column].astype(str).to_numpy()
            sid_aliases = set(alias_to_column)
            candidates = [(len({norm_gene(x) for x in values} & sid_aliases), name)
                          for name, values in options.items()]
            _, best_name = max(candidates, key=lambda item: item[0])
            if best_name != "row names" or candidates[0][0] > 0:
                marker_id_to_symbol = {
                    norm_gene(identifier): norm_gene(symbol)
                    for identifier, symbol in zip(row_names, options[best_name])
                    if pd.notna(symbol) and str(symbol).strip()
                }
                mapping_source = f"{args.imyoo_h5ad}:{best_name}"
                print(f"Translated ImYoo marker IDs using {mapping_source}")
        except Exception as exc:
            print(f"Warning: could not read ImYoo gene metadata for marker translation: {exc}")

    marker_sets = {}
    marker_text = {}
    for cluster, frame in markers.groupby("cluster", sort=False):
        genes = []
        for gene in frame.sort_values("rank")["gene"].astype(str):
            key = marker_id_to_symbol.get(norm_gene(gene), norm_gene(gene))
            if key not in genes:
                genes.append(key)
        genes = genes[:args.top_markers]
        marker_sets[str(cluster)] = genes
        marker_text[str(cluster)] = ", ".join(genes)

    scores = []
    null_scores = []
    summary = []
    rng = np.random.default_rng(0)
    sid_values = sid[gene_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Standardize each gene within each Sid cell group across draws so a score
    # compares a group's relative program over time, rather than cell-group
    # abundance or the absolute scale of either lab.
    sid_groups = sid["cell_type"].astype(str)
    means = sid_values.groupby(sid_groups, sort=False).transform("mean")
    std = sid_values.groupby(sid_groups, sort=False).transform("std", ddof=0).replace(0, 1.0).fillna(1.0)
    sid_z = (sid_values - means) / std

    for cluster, genes in marker_sets.items():
        columns = [alias_to_column[g] for g in genes if g in alias_to_column]
        used = [norm_gene(c) for c in columns]
        summary.append({
            "cluster": cluster,
            "marker_genes": marker_text[cluster],
            "genes_used": ", ".join(used),
            "n_marker_genes": len(genes),
            "n_genes_used": len(columns),
        })
        if not columns:
            continue
        rows = sid[["date", "cell_type", "n_cells"]].copy()
        rows["cluster"] = cluster
        rows["score"] = sid_z[columns].mean(axis=1).to_numpy(dtype="float32")
        rows["genes_used"] = ", ".join(used)
        rows["n_genes_used"] = len(columns)
        scores.append(rows)

        # Matched-size random gene programs provide a descriptive null. Keep
        # the same Sid rows and standardization, changing only the gene set.
        pool = [column for column in gene_columns if column not in columns]
        n_controls = min(args.null_sets, 0 if not pool else len(pool))
        for null_index in range(n_controls):
            control_columns = rng.choice(pool, size=len(columns), replace=False).tolist()
            null_rows = sid[["date", "cell_type", "n_cells"]].copy()
            null_rows["cluster"] = cluster
            null_rows["null_index"] = null_index
            null_rows["score"] = sid_z[control_columns].mean(axis=1).to_numpy(dtype="float32")
            null_scores.append(null_rows)

    if not scores:
        raise SystemExit(
            "No ImYoo cluster marker genes matched Sid pseudobulk columns. "
            "The cluster markers may use numeric H5AD IDs; ensure the ImYoo H5AD is available at "
            f"{args.imyoo_h5ad} or pass --imyoo-h5ad PATH."
        )
    result = pd.concat(scores, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    summary = pd.DataFrame(summary)
    summary["sid_rows"] = summary["cluster"].map(result.groupby("cluster").size()).fillna(0).astype(int)
    os.makedirs(args.out, exist_ok=True)
    result.to_parquet(os.path.join(args.out, "sid_cluster_scores.parquet"), index=False)
    if null_scores:
        pd.concat(null_scores, ignore_index=True).to_parquet(
            os.path.join(args.out, "sid_cluster_null_scores.parquet"), index=False)
    summary.to_parquet(os.path.join(args.out, "sid_cluster_bridge_summary.parquet"), index=False)
    info = {
        "clusters": summary["cluster"].tolist(),
        "top_markers_per_cluster": args.top_markers,
        "sid_rows": int(len(sid)),
        "sid_gene_columns": len(gene_columns),
        "clusters_with_matches": int((summary["n_genes_used"] > 0).sum()),
        "null_sets_per_cluster": args.null_sets,
        "marker_gene_mapping": mapping_source,
        "interpretation": "Scores standardized within each Sid cell group across draws for ImYoo marker programs; not transferred cell labels.",
    }
    with open(os.path.join(args.out, "sid_cluster_bridge_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
